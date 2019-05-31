"""
A neural memory module inspired by the Scene Memory Transformer paper.
"""

import typing
from collections import deque


class Memory(object):
    def __init__(self, tf_session: tf.Session, memory_size: int) -> None:
        self.session = tf_session
        self.memory_size = memory_size
        self.observations: typing.Deque[Any] = deque(maxlen=memory_size)

    def reset(self) -> None:
        self.observations.clear()
