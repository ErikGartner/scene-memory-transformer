import tensorflow as tf
import numpy as np

from memory.blocks import *


def test_intitialize_transformer():
    batch_size = 50
    embed_size = 10
    memory_size = 20
    dim_model = embed_size
    dim_ff = 128
    nbr_actions = 4

    sess = tf.Session()
    with sess:
        observations = tf.constant(
            np.random.rand(batch_size, memory_size, embed_size), dtype=tf.float32
        )
        current_obs = tf.constant(
            np.random.rand(batch_size, memory_size, embed_size), dtype=tf.float32
        )
        input_mask = tf.constant(
            np.ones((batch_size, memory_size, memory_size)), dtype=tf.float32
        )
        target_mask = tf.constant(
            np.ones((batch_size, memory_size, memory_size)), dtype=tf.float32
        )

        print(f"Embeddings (scene memory output): {observations.shape}")
        print(f"Current observations (state): {current_obs.shape}")

        enc = encoder(
            memory=observations,
            nbr_encoders=3,
            nbr_heads=2,
            dim_model=dim_model,
            dim_ff=dim_ff,
            input_mask=input_mask,
        )
        print(f"Encoded Memory (encoder output): {enc.shape}")

        dec = decoder(
            target=current_obs,
            context=enc,
            nbr_decoders=3,
            nbr_heads=2,
            dim_model=dim_model,
            dim_ff=dim_ff,
            input_mask=input_mask,
            target_mask=target_mask,
        )
        print(f"Decoded Memory (policy input): {dec.shape}")

        logits = tf.layers.dense(dec, nbr_actions)
        print(f"Logits: {logits.shape}")

        # Write graph to tensorboard log
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        sess.run(logits)
        writer = tf.summary.FileWriter("logs", sess.graph)
        writer.close()
