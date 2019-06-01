import numpy as np

from memory.memory import Memory


class Test_Memory(object):
    def test_add_embedding(self):
        memory_size = 10
        embedding_size = 5
        memory = Memory(memory_size=memory_size, embedding_size=embedding_size)

        # Test adding a single embedding
        memory.add_embedding(np.array([1, 2, 3, 4, 5]))
        expected = np.zeros((memory_size, embedding_size))
        expected[0, :] = np.array([1, 2, 3, 4, 5])
        np.testing.assert_equal(expected, memory.get_state())

        expected = np.zeros(memory_size)
        expected[0] = 1
        np.testing.assert_equal(expected, memory.get_mask())
