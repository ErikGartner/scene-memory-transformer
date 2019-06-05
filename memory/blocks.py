"""
Functions for constructing a transformer model.
"""

import tensorflow as tf
import tensorflow.contrib as tc


def scaled_dot_product_attention(
    Q: tf.Tensor,
    K: tf.Tensor,
    V: tf.Tensor,
    dim_model: int,
    mask: int = None,
    scope: str = "sdp_attention",
) -> tf.Tensor:
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1]

    with tf.variable_scope(scope):
        # Create K^T
        out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))

        # Scale by dimension
        out = out / tf.sqrt(tf.cast(dim_model, tf.float64))

        if mask is not None:
            # Set to -Inf for 0 in mask
            out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

        out = tf.nn.softmax(out)
        out = tf.matmul(out, V)

    return out


def multihead_attention(
    query: tf.Tensor,
    memory: tf.Tensor,
    nbr_heads: int,
    dim_model: int,
    mask: int = None,
    scope: str = "multihead_attention",
) -> tf.Tensor:

    if memory is None:
        memory = query

    with tf.variable_scope(scope):
        # Linear projections
        # dimensions: [batch, q_size / k_size, model_dim]
        Q = tf.layers.dense(query, dim_model, activation=tf.nn.relu)
        K = tf.layers.dense(memory, dim_model, activation=tf.nn.relu)
        V = tf.layers.dense(memory, dim_model, activation=tf.nn.relu)

        Q_split = tf.concat(tf.split(Q, nbr_heads, axis=2), axis=0)
        K_split = tf.concat(tf.split(K, nbr_heads, axis=2), axis=0)
        V_split = tf.concat(tf.split(V, nbr_heads, axis=2), axis=0)

        if mask is not None:
            mask_split = tf.tile(mask, [nbr_heads, 1, 1])
        else:
            mask_split = mask

        # Apply scaled dot product attention
        out = scaled_dot_product_attention(
            Q=Q_split,
            K=K_split,
            V=V_split,
            mask=mask_split,
            dim_model=dim_model,
        )

        # Merge the multi-head back to the original shape
        out = tf.concat(tf.split(out, nbr_heads, axis=0), axis=2)

    return out


def pointwise_feedforward(
    x: tf.Tensor,
    dim_ff: int,
    dim_model: int,
    scope: str = "pointwise_feedforward",
) -> tf.Tensor:

    out = x
    with tf.variable_scope(scope):
        out = tf.layers.conv1d(
            out, filters=dim_ff, kernel_size=1, activation=tf.nn.relu
        )
        out = tf.layers.conv1d(out, filters=dim_model, kernel_size=1)

    return out


def encoder_layer(
    x: tf.Tensor,
    nbr_heads: int,
    dim_model: int,
    dim_ff: int,
    scope: str,
    mask: tf.Tensor = None,
) -> tf.Tensor:

    out = x
    with tf.variable_scope(scope):
        out = tc.layers.layer_norm(
            out
            + multihead_attention(
                query=out,
                memory=None,
                nbr_heads=nbr_heads,
                dim_model=dim_model,
                mask=mask,
            ),
            center=True,
            scale=True,
        )
        out = tc.layers.layer_norm(
            out
            + pointwise_feedforward(x=out, dim_ff=dim_ff, dim_model=dim_model)
        )

    return out


def encoder(
    memory: tf.Tensor,
    nbr_encoders: int,
    nbr_heads: int,
    dim_model: int,
    dim_ff: int,
    input_mask: tf.Tensor = None,
    scope: str = "encoder",
) -> tf.Tensor:

    out = memory
    with tf.variable_scope(scope):
        for i in range(nbr_encoders):
            out = encoder_layer(
                x=out,
                nbr_heads=nbr_heads,
                dim_model=dim_model,
                dim_ff=dim_ff,
                mask=input_mask,
                scope=f"enc_{i}",
            )
    return out


def decoder_layer(
    target: tf.Tensor,
    context: tf.Tensor,
    nbr_heads: int,
    dim_model: int,
    dim_ff: int,
    scope: str,
    input_mask: tf.Tensor = None,
    target_mask: tf.Tensor = None,
) -> tf.Tensor:

    out = target
    with tf.variable_scope(scope):
        out = tc.layers.layer_norm(
            out
            + multihead_attention(
                query=out,
                memory=None,
                nbr_heads=nbr_heads,
                dim_model=dim_model,
                mask=target_mask,
                scope="multihead_attention_0",
            ),
            center=True,
            scale=True,
        )
        out = tc.layers.layer_norm(
            out
            + multihead_attention(
                query=out,
                memory=context,
                nbr_heads=nbr_heads,
                dim_model=dim_model,
                mask=input_mask,
                scope="multihead_attention_1",
            ),
            center=True,
            scale=True,
        )
        out = tc.layers.layer_norm(
            out
            + pointwise_feedforward(x=out, dim_ff=dim_ff, dim_model=dim_model),
            center=True,
            scale=True,
        )

    return out


def decoder(
    target: tf.Tensor,
    context: tf.Tensor,
    nbr_decoders: int,
    nbr_heads: int,
    dim_model: int,
    dim_ff: int,
    input_mask: tf.Tensor = None,
    target_mask: tf.Tensor = None,
    scope: str = "decoder",
) -> tf.Tensor:

    out = target
    with tf.variable_scope(scope):
        for i in range(nbr_decoders):
            out = decoder_layer(
                target=out,
                context=context,
                nbr_heads=nbr_heads,
                dim_model=dim_model,
                dim_ff=dim_ff,
                input_mask=input_mask,
                target_mask=target_mask,
                scope=f"dec_{i}",
            )
    return out


def create_transformer(
    observation: tf.Tensor,
    memory: tf.Tensor,
    dim_model: int,
    dim_ff: int,
    nbr_heads: int,
    nbr_encoders: int,
    nbr_decoders: int,
    input_mask: tf.Tensor = None,
    target_mask: tf.Tensor = None,
):
    enc = encoder(
        memory=memory,
        nbr_encoders=nbr_encoders,
        nbr_heads=nbr_heads,
        dim_model=dim_model,
        dim_ff=dim_ff,
        input_mask=input_mask,
    )
    dec = decoder(
        target=observation,
        context=enc,
        nbr_decoders=nbr_decoders,
        nbr_heads=nbr_heads,
        dim_model=dim_model,
        dim_ff=dim_ff,
        input_mask=input_mask,
        target_mask=target_mask,
    )
    return dec
