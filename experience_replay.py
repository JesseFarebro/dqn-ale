import numpy as np

from config import cfg


class CircularBuffer:
    def __init__(self, maxlen, shape, dtype):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.empty((maxlen,) + shape, dtype=dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0 or idx >= self.length:
                raise KeyError()
        elif isinstance(idx, np.ndarray):
            if (idx < 0).any() or (idx >= self.length).any():
                raise KeyError()
        return self.data.take(self.start + idx, mode="wrap", axis=0)

    def __array__(self):
        return self.data.take(
            np.arange(self.start, self.start + self.length), mode="wrap", axis=0
        )

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            raise RuntimeError()

        self.data[(self.start + self.length - 1) % self.maxlen] = v


class ExperienceReplay:
    def __init__(self, size, obs_shape):
        self._obs_height, self._obs_width, self._obs_channels = obs_shape

        self.size = size

        self.observations = CircularBuffer(size, obs_shape, np.uint8)
        self.actions = CircularBuffer(size, (), np.uint8)
        self.rewards = CircularBuffer(size, (), np.int32)
        self.terminals = CircularBuffer(size, (), np.bool)

    def __len__(self):
        return len(self.observations)

    def _get_full_observations(self, samples, batch_size):
        full_observation = np.empty(
            (batch_size, self._obs_height, self._obs_width, cfg.frame_history_size),
            dtype=np.uint8,
        )
        for batch_index, start_idx in enumerate(samples):
            assert start_idx >= cfg.frame_history_size - 1
            assert start_idx <= self.size - 1
            start_range_idx = start_idx - (cfg.frame_history_size - 1)
            end_range_idx = start_range_idx + cfg.frame_history_size

            frame_index_range = np.arange(start_range_idx, end_range_idx, dtype=np.int)
            terminal = np.argwhere(self.terminals[frame_index_range])
            assert len(frame_index_range) == cfg.frame_history_size
            assert frame_index_range[cfg.frame_history_size - 1] == start_idx
            assert len(terminal) <= 1

            full_observation[batch_index] = np.concatenate(
                self.observations[frame_index_range], axis=2
            )
            # Zero out frames that don't come from the current episode
            if len(terminal) > 0:
                full_observation[batch_index, :, :, : np.squeeze(terminal) + 1] = 0

        return full_observation

    def append(self, obs, action, reward, terminal):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def sample(self, batch_size):
        assert len(self.observations) >= batch_size
        samples = np.random.randint(
            cfg.frame_history_size, len(self.observations), size=batch_size
        )

        batch_observations = self._get_full_observations(
            (samples - 1) % self.size, batch_size
        )
        batch_rewards = self.rewards[samples]
        batch_actions = self.actions[samples]
        batch_next_observation = self._get_full_observations(samples, batch_size)
        batch_terminals = self.terminals[samples]

        return (
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_next_observation,
            batch_terminals,
        )
