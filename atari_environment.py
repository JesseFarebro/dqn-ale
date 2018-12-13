import numpy as np

from ale_python_interface import ALEInterface
from config import cfg
from experience_replay import CircularBuffer


class AtariEnvironment:
    num_actions = 18  # Use full action set

    def __init__(self, frame_shape, frame_postprocess=lambda x: x):
        self.ale = ALEInterface()
        self.ale.setBool(b"display_screen", cfg.display_screen)
        self.ale.setInt(b"frame_skip", 1)
        self.ale.setBool(b"color_averaging", False)
        self.ale.setInt(b"random_seed", cfg.random_seed)
        self.ale.setFloat(b"repeat_action_probability", cfg.sticky_prob)

        self.ale.loadROM(str.encode(cfg.rom))

        self.ale.setMode(cfg.mode)
        self.ale.setDifficulty(cfg.difficulty)

        self.action_set = self.ale.getLegalActionSet()
        assert len(self.action_set) == AtariEnvironment.num_actions

        screen_dims = tuple(reversed(self.ale.getScreenDims())) + (1,)
        self._frame_buffer = CircularBuffer(
            cfg.frame_buffer_size, screen_dims, np.uint8
        )
        self._frame_stack = CircularBuffer(
            cfg.frame_history_size, frame_shape, np.uint8
        )
        self._frame_postprocess = frame_postprocess

        self._episode_count = 0
        self.reset(inc_episode_count=False)

    def _is_terminal(self):
        return self.ale.game_over()

    def _get_single_frame(self):
        stacked_frames = np.concatenate(self._frame_buffer, axis=2)
        maxed_frame = np.amax(stacked_frames, axis=2)
        expanded_frame = np.expand_dims(maxed_frame, 3)
        frame = self._frame_postprocess(expanded_frame)

        return frame

    def reset(self, inc_episode_count=True):
        self._episode_frames = 0
        self._episode_reward = 0
        if inc_episode_count:
            self._episode_count += 1

        self.ale.reset_game()
        for _ in range(cfg.frame_buffer_size):
            self._frame_buffer.append(self.ale.getScreenGrayscale())
        for _ in range(cfg.frame_history_size):
            self._frame_stack.append(self._get_single_frame())

    def act(self, action):
        assert not self._is_terminal()

        cum_reward = 0
        for _ in range(cfg.frame_skip):
            cum_reward += self.ale.act(self.action_set[action])
            self._frame_buffer.append(self.ale.getScreenGrayscale())

        self._frame_stack.append(self._get_single_frame())
        self._episode_frames += cfg.frame_skip
        self._episode_reward += cum_reward
        cum_reward = np.clip(cum_reward, -1, 1)

        return cum_reward, self.state, self._is_terminal()

    @property
    def state(self):
        assert len(self._frame_buffer) == cfg.frame_buffer_size
        assert len(self._frame_stack) == cfg.frame_history_size
        return np.concatenate(self._frame_stack, axis=-1)

    @property
    def episode_reward(self):
        return self._episode_reward

    @property
    def episode_frames(self):
        return self._episode_frames

    @property
    def episode_steps(self):
        return self._episode_frames // cfg.frame_skip

    @property
    def episode_count(self):
        return self._episode_count
