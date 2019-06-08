"""
A neural memory module inspired by the Scene Memory Transformer paper.
"""

import typing
from collections import deque

import numpy as np
import tensorflow as tf


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


def update_memory(observation, memory, mask, reset):
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
    new_memory = tf.concat([tf.expand_dims(observation, 0), new_memory[:-1, :]], axis=0)

    # Update mask
    new_mask = mask * (1 - reset)
    new_mask = tf.concat([tf.ones((1)), new_mask[:-1]], axis=0)
    return new_memory, new_mask


def batch_update_memory(observations, start_memory, start_mask, dones_ph):
    """
    Takes a number of observations in batch and creates appropriate memory
    and mask for each observation.
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
    for batch_idx in range(nbr_obs):
        new_mem, new_mask = update_memory(
            tf.squeeze(obs[batch_idx]), new_mem, new_mask, tf.squeeze(dones[batch_idx])
        )
        masks.append(new_mask)
        memories.append(new_mem)

    batch_memory = tf.stack(memories, axis=0)
    batch_mask = tf.stack(masks, axis=0)
    assert (
        batch_memory.shape[0] == dones_ph.shape[0]
        and batch_memory.shape[1] == start_memory.shape[0]
        and batch_memory.shape[2] == start_memory.shape[1]
    ), f"Incorrect memory output shape: {batch_memory.shape}"
    assert (
        batch_mask.shape[0] == dones_ph.shape[0]
        and batch_mask.shape[1] == start_mask.shape[0]
    ), f"Incorrect mask output shape: {batch_mask.shape}"
    return batch_memory, batch_mask


def empty_state(memory_size, embed_size):
    """Returns an empty state for the memory"""
    return (
        tf.zeros((memory_size, embed_size), dtype=np.float32),
        tf.zeros((memory_size), dtype=np.float32),
    )
