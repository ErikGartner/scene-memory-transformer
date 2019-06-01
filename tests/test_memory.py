import numpy as np

from memory.memory import Memory


class Test_Memory(object):
    def setup(self):
        self.memory_size = 10
        self.embedding_size = 5
        self.memory = Memory(
            memory_size=self.memory_size, embedding_size=self.embedding_size
        )

    def test_add_embedding(self):

        # Test adding a single embedding
        self.memory.add_embedding(np.array([1, 2, 3, 4, 5]))
        expected = np.zeros((self.memory_size, self.embedding_size))
        expected[0, :] = np.array([1, 2, 3, 4, 5])
        np.testing.assert_equal(expected, self.memory.get_state())

        expected = np.zeros(self.memory_size)
        expected[0] = 1
        np.testing.assert_equal(expected, self.memory.get_mask())

        assert len(self.memory) == 1

    def test_set_state(self):
        state = np.random.rand(self.memory_size, self.embedding_size)
        mask = np.ones(self.memory_size)

        self.memory.set_state(state, mask)
        print(self.memory.get_state())
        print(self.memory.embeddings)
