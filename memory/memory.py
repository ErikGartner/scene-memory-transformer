"""
A neural memory module inspired by the Scene Memory Transformer paper.
"""

import typing


class Memory(object):
    def __init__(self, tf_session, nbr_encoders: int = 6) -> None:
        self.session = tf_session
