import numpy as np

import pytest
from experience_replay import ExperienceReplay


@pytest.mark.incremental
class TestExperienceReplay:
    def test_observation_construction(self):
        """ Tests observation construction from partial observations """
        obs_shape = (84, 84, 1)
        er = ExperienceReplay(5, obs_shape)

        obs_ = []
        obs_next_ = []
        for i in range(1, 6):
            partial_obs = np.ones(obs_shape) * i
            if i < 5:
                obs_.append(partial_obs)
            if i > 1:
                obs_next_.append(partial_obs)
            er.append(partial_obs, 0, 0, 0)
        obs_ = np.transpose(np.array(obs_), (3, 1, 2, 0))
        obs_next_ = np.transpose(np.array(obs_next_), (3, 1, 2, 0))

        batch = er.sample(1)
        obs, rewards, actions, obs_next, terminals = batch
        assert np.array_equal(obs_, obs)
        assert np.array_equal(obs_next_, obs_next)

    def test_observation_zeroing(self):
        """ Tests zeroing out of frames not from current episode """
        obs_shape = (84, 84, 1)
        er = ExperienceReplay(5, obs_shape)

        for terminal_idx in range(5):
            obs_ = []
            obs_next_ = []
            for i in range(1, 6):
                partial_obs = np.ones(obs_shape) * i
                terminal = 1 if i == terminal_idx else 0
                er.append(partial_obs, 0, 0, terminal)

                if i <= terminal_idx:
                    partial_obs *= 0
                if i < 5:
                    obs_.append(partial_obs)
                if i > 1:
                    obs_next_.append(partial_obs)
            obs_ = np.transpose(np.array(obs_), (3, 1, 2, 0))
            obs_next_ = np.transpose(np.array(obs_next_), (3, 1, 2, 0))

            batch = er.sample(1)
            obs, rewards, actions, obs_next, terminals = batch
            assert np.array_equal(obs_, obs)
            assert np.array_equal(obs_next_, obs_next)

    def test_sampling(self):
        """ Tests observation construction from partial observations """
        obs_shape = (84, 84, 1)
        er = ExperienceReplay(5, obs_shape)

        for i in range(1, 6):
            partial_obs = np.ones(obs_shape) * i
            er.append(partial_obs, 1, 1, 0)

        batch = er.sample(1)
        _, rewards, actions, _, terminals = batch
        assert np.array_equal(rewards, np.array([1]))
        assert np.array_equal(actions, np.array([1]))
        assert np.array_equal(terminals, np.array([0]))
