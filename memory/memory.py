"""
A neural memory module inspired by the Scene Memory Transformer paper.
"""

import typing
from collections import deque

import numpy as np


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

    def get_mask(self):
        return np.array(self.input_mask)

    def add_embedding(self, emb: np.ndarray):
        """Adds an embedding to the memory and update the input mask.

        :param np.ndarray emb: The new embedding.
        """
        emb = emb.squeeze()
        assert emb.shape == (self.embedding_size,)
        self.embeddings.appendleft(emb)
        size = len(self.embeddings)
        self.input_mask[size - 1] = 1

    def __len__(self):
        return len(self.embeddings)
