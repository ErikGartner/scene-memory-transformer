import tensorflow as tf
import numpy as np

from memory.blocks import *


def test_intitialize_encoder():
    sess = tf.Session()
    with sess:
        val = np.random.rand(50, 10, 10)  # batch size, seq len, embed_size
        x = tf.constant(val)
        enc = encoder(x, nbr_encoders=3, nbr_heads=2, dim_model=10, dim_ff=20)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        print(sess.run(enc))
        writer = tf.summary.FileWriter("logs", sess.graph)
        writer.close()
