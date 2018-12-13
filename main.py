import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from atari_environment import AtariEnvironment
from config import cfg
from dqn import DQN
from experience_replay import CircularBuffer, ExperienceReplay


def main(_):
    # Reproducability
    tf.reset_default_graph()
    np.random.seed(cfg.random_seed)
    tf.set_random_seed(cfg.random_seed)

    # Logging
    summary_writer = tf.contrib.summary.create_file_writer(cfg.log_dir)
    summary_writer.set_as_default()

    episode_ph = tf.placeholder(tf.int64, (), name="episode")
    reward_ph = tf.placeholder(tf.float32, (), name="epeisode/reward")
    step_ph = tf.placeholder(tf.int64, (), name="episode/steps")

    with tf.contrib.summary.always_record_summaries():
        episode_summary = [
            tf.contrib.summary.scalar("episode/reward", reward_ph, step=episode_ph),
            tf.contrib.summary.scalar("episode/step", step_ph, step=episode_ph),
        ]

    if not tf.gfile.Exists(cfg.save_dir):
        tf.gfile.MakeDirs(cfg.save_dir)

    episode_results_path = os.path.join(cfg.save_dir, "episodeResults.csv")
    episode_results = tf.gfile.GFile(episode_results_path, "w")
    episode_results.write("episode,reward,steps\n")

    # Setup ALE and DQN graph
    obs_shape = (84, 84, 1)
    input_height, input_width, _ = obs_shape

    # Log DQN summaries every n steps
    with tf.contrib.summary.record_summaries_every_n_global_steps(
        cfg.log_summary_every
    ):
        dqn = DQN(input_height, input_width, AtariEnvironment.num_actions)

    # Global step
    global_step = tf.train.get_or_create_global_step()
    increment_step = tf.assign_add(global_step, 1)

    # Save all variables
    vars_to_save = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=cfg.q_scope
    )
    saver = tf.train.Saver(var_list=vars_to_save)

    # Handle loading specific variables
    restoring = cfg.restore_dir is not None
    vars_to_load = []
    if restoring:
        for scope in cfg.load_scope.split(","):
            vars_to_load.extend(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            )
        loader = tf.train.Saver(var_list=vars_to_load)

    tf.logging.info("Variables to save: ")
    tf.logging.info(vars_to_save)
    tf.logging.info("Variables to load: ")
    tf.logging.info(vars_to_load)

    # Setup session
    def init_fn(scaffold, sess):
        tf.contrib.summary.initialize(session=sess)
        if restoring:
            chkpt = (
                tf.train.latest_checkpoint(cfg.restore_dir)
                if cfg.restore_file is None
                else os.path.join(cfg.restore_dir, cfg.restore_file)
            )
            tf.logging.info("Restoring weights from checkpoint %s" % chkpt)
            loader.restore(sess, chkpt)

    scaffold = tf.train.Scaffold(init_fn=init_fn)
    sess = tf.train.SingularMonitoredSession(scaffold=scaffold)
    sess.run(dqn.copy_to_target)

    # Initialize ALE
    postprocess_frame = lambda frame: sess.run(
        dqn.process_frame, feed_dict={dqn.image: frame}
    )
    env = AtariEnvironment(obs_shape, postprocess_frame)

    # Replay buffer
    replay_buffer = ExperienceReplay(cfg.replay_buffer_size, obs_shape)

    # Perform random policy to get some training data
    with tqdm(
        total=cfg.seed_frames, disable=cfg.disable_progress or cfg.evaluate
    ) as pbar:
        seed_steps = 0
        while seed_steps * cfg.frame_skip < cfg.seed_frames and (
            not sess.should_stop() and not cfg.evaluate
        ):
            action = np.random.randint(AtariEnvironment.num_actions)
            reward, next_state, terminal = env.act(action)
            seed_steps += 1

            replay_buffer.append(
                next_state[:, :, -1, np.newaxis], action, reward, terminal
            )

            if terminal:
                pbar.update(env.episode_frames)
                env.reset(inc_episode_count=False)

    if cfg.evaluate:
        assert cfg.max_episode_count > 0
    else:
        assert len(replay_buffer) >= cfg.seed_frames // cfg.frame_skip

    # Main training loop
    steps = tf.train.global_step(sess, global_step)
    env.reset(inc_episode_count=False)
    terminal = False

    total = cfg.max_episode_count if cfg.evaluate else cfg.num_frames
    with tqdm(total=total, disable=cfg.disable_progress) as pbar:
        # Loop while we haven't observed our max frame number
        # If we are at our max frame number we will finish the current episode
        while (
            not (
                # We must be evaluating or observed the last frame
                # As well as be terminal
                # As well as seen the maximum episode number
                (steps * cfg.frame_skip > cfg.num_frames or cfg.evaluate)
                and terminal
                and env.episode_count >= cfg.max_episode_count
            )
            and not sess.should_stop()
        ):
            # Epsilon greedy policy with epsilon annealing
            if not cfg.evaluate and steps * cfg.frame_skip < cfg.eps_anneal_over:
                # Only compute epsilon step while we're still annealing epsilon
                epsilon = cfg.eps_initial - steps * (
                    (cfg.eps_initial - cfg.eps_final) / cfg.eps_anneal_over
                )
            else:
                epsilon = cfg.eps_final

            # Epsilon greedy policy
            if np.random.uniform() < epsilon:
                action = np.random.randint(0, AtariEnvironment.num_actions)
            else:
                action = sess.run(dqn.action, feed_dict={dqn.S: [env.state]})

            # Perform environment step
            steps = sess.run(increment_step)
            reward, next_state, terminal = env.act(action)

            if not cfg.evaluate:
                replay_buffer.append(
                    next_state[:, :, -1, np.newaxis], action, reward, terminal
                )

                # Sample and do gradient updates
                if steps % cfg.learning_freq == 0:
                    placeholders = [
                        dqn.S,
                        dqn.actions,
                        dqn.rewards,
                        dqn.S_p,
                        dqn.terminals,
                    ]
                    batch = replay_buffer.sample(cfg.batch_size)
                    sess.run(
                        [dqn.train, dqn.summary],
                        feed_dict=dict(zip(placeholders, batch)),
                    )
                if steps % cfg.target_update_every == 0:
                    sess.run([dqn.copy_to_target])
                if steps % cfg.model_chkpt_every == 0:
                    saver.save(
                        sess.raw_session(), "%s/model_epoch_%04d" % (cfg.log_dir, steps)
                    )

            if terminal:
                episode_results.write(
                    "%d,%d,%d\n"
                    % (env.episode_count, env.episode_reward, env.episode_frames)
                )
                episode_results.flush()
                # Log episode summaries to Tensorboard
                sess.run(
                    episode_summary,
                    feed_dict={
                        reward_ph: env.episode_reward,
                        step_ph: env.episode_frames // cfg.frame_skip,
                        episode_ph: env.episode_count,
                    },
                )
                pbar.update(env.episode_frames if not cfg.evaluate else 1)
                env.reset()

    episode_results.close()
    tf.logging.info(
        "Finished %d %s"
        % (
            cfg.max_episode_count if cfg.evaluate else cfg.num_frames,
            "episodes" if cfg.evaluate else "frames",
        )
    )


if __name__ == "__main__":
    tf.app.run()
