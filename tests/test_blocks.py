import tensorflow as tf
import numpy as np

from memory.blocks import *


def test_intitialize_transformer():
    batch_size = 50
    embed_size = 10
    seq_len = 20

    sess = tf.Session()
    with sess:
        observations = tf.constant(np.random.rand(batch_size, seq_len, embed_size))
        current_obs = tf.constant(np.random.rand(batch_size, 1, embed_size))

        print(f"Embeddings (scene memory output): {observations.shape}")
        print(f"Current observations (state): {current_obs.shape}")

        enc = encoder(
            observations, nbr_encoders=3, nbr_heads=2, dim_model=10, dim_ff=20
        )
        print(f"Encoded Memory (encoder output): {enc.shape}")

        dec = decoder(
            enc, observations, nbr_decoders=3, nbr_heads=2, dim_model=10, dim_ff=20
        )
        print(f"Decoded Memory (policy input): {dec.shape}")

        # Write graph to tensorboard log
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        # print(sess.run(dec))
        writer = tf.summary.FileWriter("logs", sess.graph)
        writer.close()
