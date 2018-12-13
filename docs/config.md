# Configuration

It is very important to note the difference between different time scales which are used in the configuration of DQN.

There are _steps_ which are defined as the number of times the agent has interacted with the environment.

We also talk about frames, which is defined as `steps * frame_skip` where `frame_skip` is the number of frames
the environment fast forwards between actions.

# Global Configuration

## `random_seed`
### Type: `integer`
### Default: `0`
Used to seed TensorFlow and numpy's random number generator. Used to reproduce results.

## `evaluate`
### Type: `boolean`
### Default: `False`
Evaluation boolean, used to disable dropout, epsilon annealing, training, and seeding the replay buffer. This is useful to evalaute a checkpoint on a different environment.

# Environment Configuration

## `sticky_prob`
### Type: `float`
### Default: `0.25`
Sticky action probability. With probability `sticky_prob` the environment will exeucte the previous action instead of the current action. See [1].

## `mode`
### Type: `integer`
### Default: `0`
Game mode for the specific ROM. It's best to look up game modes and difficulties in the manual for the game you are running.

## `difficulty`
### Type: `integer`
### Default: `0`
Difficulty for the specific ROM. It's best to look up game modes and difficulties in the manual for the game you are running. Difficulty is independent of the mode, not every mode will support every difficulty. Consult the game manual for your game.

## `frame_skip`
### Type: `integer`
### Default: `5`
How many frames should the environment skip before returning the next observation. When the agent executes an action `a` the environment will fast foward `frame_skip` frames before returning the `frame_skip + 1`-th frame as the observation. Previous implementations used `4` but following the advice of [1] we use `5`.

## `frame_buffer_size`
### Type: `integer`
### Default: `2`
We take the maximum over the last `frame_buffer_size` frames to prevent flickering of objects in some games.

## `display_screen`
### Type: `boolean`
### Default: `False`
Display the game screen for the user to monitor. This will significantly slow down training. Use for debugging.

# RL Specific Configuration

## `gamma`
### Type: `float`
### Default: `0.99`
Discount rate for the agent, typical will be close to `0.99`.

## `eps_initial`
### Type: `float`
### Default: `1.0`
Initial epsilon value for the epsilon-greedy policy. The value is decayed over `eps_anneal_over` frames.

## `eps_final`
### Type: `float`
### Default: `0.01`
Final epsilon value for the epsilon-greedy policy. The value will have been annealed over `eps_anneal_over` frames.

## `eps_anneal_over`
### Type: `integer`
### Default: `1 000 000`
How many frames to linnearly anneal epsilon over. See `eps_initial`, `eps_final`.


# DQN Configuration

## `replay_buffer_size`
### Type: `integer`
### Default: `1 000 000`
How many observations should the replay buffer hold. Defaults to 1 million observations. See [2] for a more in depth analysis of the replay buffer.

## `learning_freq`
### Type: `integer`
### Default: `4`
How many agent steps before we update our Q networks.


## `frame_history_size`
### Type: `integer`
### Default: `4`
How many frames should be concatenated annd given as input to the agent. This was described as giving the agent a sense of velocity and acceleration in games like Breakout.


## `batch_size`
### Type: `integer`
### Default: `32`
Batch size used when performing Q-network updates.


## `target_update_every`
### Type: `integer`
### Default: `40 000`
How often to update the target network. This is in terms of agent steps, which is when the agent takes an action in the environment.


## `seed_frames`
### Type: `integer`
### Default: `50 000`
How many frames should we use to seed the replay buffer before commencing learning. We execute a random policy to generate these frames.
Note that this is in term of frames and not steps.


## `pad_first_conv_layer`
### Type: `boolean`
### Default: `False`
Add padding to the first convolutional layer. DeepMind does this in their Nature paper. This shouldn't be necesarry, unless using `padding=same` on convolutional layers.


## `q_scope`
### Type: `string`
### Default: `q`
TensorFlow variable scope to scope the Q-network. This can be useful if we are loading from a saved model or if we need to specify specific layers to restore.


## `q_target_scope`
### Type: `string`
### Default: `target`
TensorFlow variable scope to scope the target Q-network. Note that `tf.get_collection` uses regular expressions to find variables. For example if you have `q_target_scope = "q_target"` and `q_scope = "q"` we will be saving the target network and restoring the target network as `q_target` will match for the regular expression `q`.


## `learning_rate`
### Type: `float`
### Default: `0.00025`
Learning rate to be used by RMSProp. The default is a fairly standard learning rate for training DQN on the Atari 2600 games.


## `rmsprop_decay`
### Type: `float`
### Default: `0.95`
RMSProp decay rate to be used when decaying past gradients durinngn the RMSProp update.


## `rmsprop_epsilon`
### Type: `float`
### Default: `0.01`
RMSProp epsilon to prevent divide by zero errors.


# Experiment Configuration

## `num_frames`
### Type: `integer`
### Default: `50 000 000`
Total number of frames to run for. Note your experiment might run more than `num_frames` frames as we finish the last episode even if the counter exceeds `num_frames`.

## `max_episode_count`
### Type: `integer`
### Default: `-1`
Max number of episodes to run for. If the number of episodes exceeds `max_episode_count` the main routine will terminate regardless of `num_frames`. Useful for evaluation.
If `max_episode_count` is specified the routine will run `max_episode_count` episodes regardless of `num_frames`. It could be the case where `frames > num_frames` because the episode count hasn't exceeded `max_episode_counnt`.


## `restore_dir`
### Type: `string`
### Default: ``
Directory used to restore the _LAST_ checkpoint. If `restore_dir` is specified we will attempt to resetore the parameters scoped by `load_scope`.


## `restore_file`
### Type: `string`
### Default: ``
Specific file used to restore the variables scoped by `load_scope`. We load the file `restore_dir/restore_file` so `restore_file` must be in `restore_dir`.


## `save_dir`
### Type: `string`
### Default: `results`
Directory to save results for future processing. Currently it saves episode results in a csv format.


## `log_dir`
### Type: `string`
### Default: `logdir`
Where to save TensorBoard summaries and model checkpoints.


## `model_chkpt_every`
### Type: `integer`
### Default: `1 000 000`
How many frames before we save a model checkpoint. Checkpoints are saved in `log_dir`.


## `log_summary_every`
### Type: `integer`
### Default: `32`
How often to log TensorBoard summaries, currently logs: average Q-values, TD-error, optimization loss.


## `load_scope`
### Type: `string`
### Default: `None`
Scope to use while loading variables from a checkpoint. There can be multiple scopes provided by a comma delimited string, e.g., `q/conv/conv1,q/action_values/fc1`. If `None` it restores every variable.


## `optimize_scope`
### Type: `string`
### Default: `None`
Scope to use for determining which variables to optimize. There can be multiple scopes provided by a comma delimited string. If `None` we optimize every trainable variable in `q_scope`.

# Regularization Configuration

## `use_dropout`
### Type: `boolean`
### Default: `False`
Use dropout in the Q-network. Improves generalization and adaptability when used on new tasks. Used in [3].


## `use_l2`
### Type: `boolean`
### Default: `False`
Use weight decay in the Q-network. Improves generalization and adaptability when used on new tasks. Used in [3].


## `conv_dropout_rate`
### Type: `float`
### Default: `0.05`
Dropout rate used in convolutional layers. Used in [3].


## `fc_dropout_rate`
### Type: `float`
### Default: `0.1`
Dropout rate used in fully connected layers. Used in [3].


## `weight_decay_rate`
### Type: `float`
### Default: `0.001`
Weight decay rate used during optimization. Used in [3].

# References

[1] [Machado et al., Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents](https://arxiv.org/abs/1709.06009)  
[2] [Zhang et al., A Deeper Look at Experience Replay](https://arxiv.org/abs/1712.01275)  
[3] [Farebrother et al., Generalization and Regularization in DQN](https://arxiv.orrg/abs/1810.00123)  
