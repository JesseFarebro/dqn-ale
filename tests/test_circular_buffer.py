import numpy as np

import pytest
from experience_replay import CircularBuffer


@pytest.mark.incremental
class TestCircularBuffer:
    def test_append(self):
        self.buffer = CircularBuffer(5, (), np.int32)
        for i in range(5):
            self.buffer.append(i)
        assert np.array_equal(self.buffer.__array__(), np.array(range(5)))
        assert len(self.buffer) == 5

        self.buffer.append(5)
        assert np.array_equal(self.buffer.__array__(), np.array(range(1, 6)))
        assert len(self.buffer) == 5

        for i in range(5, 10):
            self.buffer.append(i)
        assert np.array_equal(self.buffer.__array__(), np.array(range(5, 10)))
        assert len(self.buffer) == 5

    def test_get(self):
        self.buffer = CircularBuffer(5, (), np.int32)
        for i in range(10):
            self.buffer.append(i)
        print(self.buffer.__array__())
        assert np.array_equal(self.buffer.__array__(), np.array(range(5, 10)))
        assert len(self.buffer) == 5

        for idx, i in enumerate(range(5, 10)):
            assert self.buffer[idx] == i
