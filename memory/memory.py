"""
A neural memory module inspired by the Scene Memory Transformer paper.
"""

import typing
from collections import deque

import numpy as np
import tensorflow as tf

"""
The numpy / python implementation of a circular buffer / memory
"""


class Memory(object):
    def __init__(self, memory_size: int, embedding_size: int) -> None:
        self.memory_size = memory_size
        self.embedding_size = embedding_size
        self.embeddings: typing.Deque[Any] = deque(maxlen=memory_size)
        self.input_mask = np.zeros(self.memory_size)

        self.reset()

    def reset(self) -> None:
        self.embeddings.clear()
        self.input_mask = np.zeros(self.memory_size)

    def set_state(self, state: np.ndarray, input_mask: np.ndarray):
        if state is None:
            self.reset()
            return

        assert input_mask.shape == (self.memory_size,)
        assert state.shape == (self.memory_size, self.embedding_size)

        size = np.count_nonzero(input_mask)
        self.embeddings = deque(state[:size, :], maxlen=self.memory_size)
        self.input_mask = np.array(input_mask)

    def get_state(self):
        size = len(self.embeddings)
        state = np.zeros((self.memory_size, self.embedding_size))
        if size > 0:
            state[:size, :] = np.array(self.embeddings)
        return state

    def get_statemask(self):
        """Returns a combined state and mask in one"""
        size = len(self.embeddings)
        state = np.zeros((self.memory_size, self.embedding_size + 1))
        state[:size, :-1] = np.array(self.embeddings)
        state[:, -1] = self.input_mask
        return state

    def set_statemask(self, statemask):
        self.set_state(statemask[:, :-1].squeeze(), statemask[:, -1].flatten())

    def get_mask(self):
        return np.array(self.input_mask)

    def add_embedding(self, emb: np.ndarray):
        """Adds an embedding to the memory and update the input mask.

        :param np.ndarray emb: The new embedding.
        """
        emb = emb.squeeze()
        assert emb.shape == (
            self.embedding_size,
        ), f"{emb.shape} vs {(self.embedding_size,)}"
        self.embeddings.appendleft(emb)
        size = len(self.embeddings)
        self.input_mask[size - 1] = 1

    def __len__(self):
        return len(self.embeddings)


"""
The Tensorflow implementation of the memory
"""


def update_memory(
    observation: tf.Tensor, memory: tf.Tensor, mask: tf.Tensor, reset: tf.Tensor
):
    """
    Update the memory and mask based on latest observation
    """
    assert (
        observation.shape[0] == memory.shape[1]
    ), f"Embedding sizes don't match, {observation.shape[0]} vs {memory.shape[1]}"
    assert mask.shape[0] == memory.shape[0], f"Memory sizes don't match"
    assert len(reset.shape.as_list()) == 0, f"Reset must be scalar"

    reset = tf.cast(reset, dtype=tf.float32)

    # Reset memory if requested
    new_memory = memory * (1 - reset)

    # Shift memory forward and add new observation
    new_memory = tf.concat(
        [tf.expand_dims(observation, 0), new_memory[:-1, :]], axis=0
    )

    # Update mask
    new_mask = mask * (1 - reset)
    new_mask = tf.concat([tf.ones((1)), new_mask[:-1]], axis=0)
    return new_memory, new_mask


def sequence_update_memory(
    observations: tf.Tensor,
    start_memory: tf.Tensor,
    start_mask: tf.Tensor,
    dones_ph: tf.Tensor,
):
    """Takes a number of observations in a sequence and creates appropriate
    memory and mask for each observation.

    :param tf.Tensor observations: Shape: (sequence_size, embedding_size)
    :param tf.Tensor start_memory: Shape: (memory_size, embedding_size)
    :param tf.Tensor start_mask: Shape: (memory_size)
    :param tf.Tensor dones_ph: Shape: (sequence_size)
    :return: Returns a tuple of the new memory, mask and the new state.
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)

    """
    assert (
        observations.shape[0] == dones_ph.shape[0]
    ), f"Done and observations do not match."

    nbr_obs = observations.shape.as_list()[0]

    masks = []
    memories = []
    obs = tf.split(observations, nbr_obs, axis=0)
    dones = tf.split(dones_ph, nbr_obs, axis=0)
    new_mem = start_memory
    new_mask = start_mask
    for seq_idx in range(nbr_obs):
        new_mem, new_mask = update_memory(
            tf.squeeze(obs[seq_idx]),
            new_mem,
            new_mask,
            tf.squeeze(dones[seq_idx]),
        )
        masks.append(new_mask)
        memories.append(new_mem)

    new_state = tf.expand_dims(
        tf.concat(
            [new_mem, tf.expand_dims(new_mask, axis=1)],
            axis=1,
            name="new_state",
        ),
        axis=0,
    )
    sequence_memory = tf.stack(memories, axis=0, name="sequence_memory")
    sequence_mask = tf.stack(masks, axis=0, name="sequence_mask")
    assert sequence_memory.shape == tf.TensorShape(
        [dones_ph.shape[0], start_memory.shape[0], start_memory.shape[1]]
    ), f"Incorrect memory output shape: {sequence_memory.shape}"
    assert sequence_mask.shape == tf.TensorShape(
        [dones_ph.shape[0], start_mask.shape[0]]
    ), f"Incorrect mask output shape: {sequence_mask.shape}"
    assert new_state.shape == tf.TensorShape(
        [
            tf.Dimension(1),
            start_memory.shape[0],
            start_memory.shape[1] + tf.Dimension(1),
        ]
    ), f"Incorrect new_state output shape: {new_state.shape}"
    return sequence_memory, sequence_mask, new_state


def batch_update_memory(
    observations: tf.Tensor,
    start_memory: tf.Tensor,
    start_mask: tf.Tensor,
    dones_ph: tf.Tensor,
):
    """Takes a batch of sequences and updates their memories.

    :param tf.Tensor observations: Shape: (batch, sequence_size, embedding_size)
    :param tf.Tensor start_memory: Shape: (batch, memory_size, embedding_size)
    :param tf.Tensor start_mask: Shape: (batch, memory_size)
    :param tf.Tensor dones_ph: Shape: (batch, sequence_size)
    :return: Returns a tuple of the new memory, mask and the new state.
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)

    """
    assert (
        observations.shape.ndims == 3
        and start_memory.shape.ndims == 3
        and start_mask.shape.ndims == 2
        and dones_ph.shape.ndims == 2
    ), "Incorrect ranks of input data"
    assert (
        observations.shape[0]
        == dones_ph.shape[0]
        == start_mask.shape[0]
        == start_memory.shape[0]
    ), "Batch size should agree for all inputs."
    assert (
        observations.shape[-1] == start_memory.shape[-1]
    ), "Embedding sizes should agreee"
    assert start_memory.shape[1] == start_mask[1], "Memory sizes should agree"
    assert (
        dones_ph.shape[-1] == observations.shape[-1]
    ), "Sequence sizes should agree"

    batch_size = observations.shape.as_list()[0]

    masks = []
    memories = []
    new_states = []
    for batch_idx in range(batch_size):
        with tf.variable_scope(f"batch_memory_{batch_idx}"):
            new_mem, new_mask, new_state = sequence_update_memory(
                observations[batch_idx, :, :],
                start_memory[batch_idx, :, :],
                start_mask[batch_idx, :],
                dones_ph[batch_idx, :],
            )
        masks.append(new_mask)
        memories.append(new_mem)
        new_states.append(new_state)

    batch_memory = tf.stack(memories, axis=0, name="batch_memory")
    batch_mask = tf.stack(masks, axis=0, name="batch_mask")
    batch_new_state = tf.stack(new_states, axis=0, name="batch_new_state")
    return batch_memory, batch_mask, batch_new_state


def empty_state(memory_size, embed_size):
    """Returns an empty state for the memory"""
    return (
        tf.zeros((memory_size, embed_size), dtype=np.float32),
        tf.zeros((memory_size), dtype=np.float32),
    )
