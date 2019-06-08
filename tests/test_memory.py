import numpy as np
import tensorflow as tf

from memory.memory import (
    Memory,
    update_memory,
    batch_update_memory,
    sequence_update_memory,
)


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
        np.testing.assert_equal(state, self.memory.get_state())
        np.testing.assert_equal(mask, self.memory.get_mask())


def test_update_memory():
    batch_size = 10
    memory_size = 20
    embed_size = 5

    sess = tf.Session()
    with sess:
        observations = tf.constant(
            np.tile(np.arange(1, batch_size + 1), (embed_size, 1)).T, dtype=tf.float32
        )
        memory = tf.constant(np.zeros((memory_size, embed_size), dtype=np.float32))
        mask = tf.constant(np.zeros(memory_size, dtype=np.float32))
        done_ph = tf.constant(np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0]))

        new_mem = memory
        new_mask = mask
        for idx in range(batch_size):

            new_mem, new_mask = update_memory(
                tf.squeeze(observations[idx, :]),
                new_mem,
                new_mask,
                tf.squeeze(done_ph[idx]),
            )

        batch_memory, batch_mask, new_state = sequence_update_memory(
            observations, memory, mask, done_ph
        )

        mem_np, mask_np = sess.run([batch_memory, batch_mask])

        corr_mem = np.zeros((3, memory_size, embed_size))
        corr_mem[0, 0, :] = np.ones(embed_size)
        corr_mem[1, 1, :] = np.ones(embed_size)
        corr_mem[2, 2, :] = np.ones(embed_size)
        corr_mem[1, 0, :] = np.ones(embed_size) * 2
        corr_mem[2, 1, :] = np.ones(embed_size) * 2
        corr_mem[2, 0, :] = np.ones(embed_size) * 3
        np.testing.assert_equal(mem_np[0:3, :, :], corr_mem)

        corr_mask = np.zeros((3, memory_size))
        corr_mask[0, 0] = 1
        corr_mask[1, 0:2] = 1
        corr_mask[2, 0:3] = 1
        np.testing.assert_equal(mask_np[0:3, :], corr_mask)
