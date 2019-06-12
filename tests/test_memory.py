import time

import numpy as np
import tensorflow as tf
import pytest

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
            np.tile(np.arange(1, batch_size + 1), (embed_size, 1)).T,
            dtype=tf.float32,
        )
        memory = tf.constant(
            np.zeros((memory_size, embed_size), dtype=np.float32)
        )
        mask = tf.constant(np.zeros(memory_size, dtype=np.float32))
        done_ph = tf.constant(np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0]))

        # Test the function
        new_mem = memory
        new_mask = mask
        for idx in range(batch_size):

            new_mem, new_mask = update_memory(
                tf.squeeze(observations[idx, :]),
                new_mem,
                new_mask,
                tf.squeeze(done_ph[idx]),
            )

        # Test the sequence version
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


tries = 5
batch_sizes = np.random.randint(1, 5, tries)
memory_sizes = np.random.randint(1, 20, tries)
sequence_lengths = np.random.randint(1, 20, tries)
test_params = [
    (int(batch_sizes[x]), int(memory_sizes[x]), int(sequence_lengths[x]))
    for x in range(tries)
]


@pytest.mark.parametrize("batch_size,memory_size,sequence_length", test_params)
def test_compare_implementations(batch_size, memory_size, sequence_length):
    """Compares the results of the TF and NP implementation of the SMT memory"""
    with tf.Session() as sess:

        embed_size = 10

        # We have one np memory for each batch
        memories = [
            Memory(memory_size=memory_size, embedding_size=embed_size)
            for _ in range(batch_size)
        ]

        # We create a start memory and mask for all batches
        memory_tf = tf.zeros(
            (batch_size, memory_size, embed_size), dtype=tf.float32
        )
        mask_tf = tf.zeros((batch_size, memory_size), dtype=tf.float32)
        done_np = np.random.choice(
            [0, 1], (batch_size, sequence_length), True, p=[0.9, 0.1]
        )
        done_tf = tf.constant(done_np, dtype=tf.float32)

        total_obs = []
        for seq_idx in range(sequence_length):

            batch_obs = []

            # Batches are parallel memories for parallel environments
            for batch_idx in range(batch_size):
                # Generate random observation
                obs_np = np.array(np.random.rand(embed_size), dtype=np.float32)
                batch_obs.append(tf.constant(obs_np, dtype=tf.float32))

                if done_np[batch_idx, seq_idx] == 1:
                    # if done, reset memory
                    memories[batch_idx].reset()

                # Add to memory
                memories[batch_idx].add_embedding(obs_np)

            # Gather observations into batches of sequences
            batch_obs_tf = tf.stack(batch_obs)  # (batch, embed)
            total_obs.append(batch_obs_tf)

        input_obs = tf.stack(total_obs, axis=1)  # (batch, seq, embed)
        batch_memory, batch_mask, batch_new_state = batch_update_memory(
            input_obs, memory_tf, mask_tf, done_tf
        )

        # Verify outputs
        tf_res_memory, tf_res_mask, tf_res_new_state = sess.run(
            [batch_memory, batch_mask, batch_new_state]
        )

        # Compare results
        for batch_idx, memory in enumerate(memories):
            np.testing.assert_array_equal(
                tf_res_memory[batch_idx, -1, :, :],
                memory.get_state(),
                "Incorrect memory after batch update",
            )
            np.testing.assert_array_equal(
                tf_res_mask[batch_idx, -1, :],
                memory.get_mask(),
                "Incorrect mask after batch update",
            )
            np.testing.assert_array_equal(
                np.squeeze(tf_res_new_state[batch_idx, :, :], axis=0),
                memory.get_statemask(),
                "Incorrect new state after batch update",
            )
