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
    mask: tf.Tensor = None,
    scope: str = "sdp_attention",
) -> tf.Tensor:
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1]

    with tf.variable_scope(scope):
        # Create K^T
        out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]), name="Q_mult_K")

        # Scale by dimension
        factor = Q.shape.as_list()[-1]
        out = tf.divide(
            out, tf.sqrt(tf.cast(factor, tf.float32)), name="Q_mult_K_scaled"
        )

        if mask is not None:
            # Set to -Inf for 0 in mask
            assert (
                out.shape == mask.shape
            ), f"Incorrect mask dimensions: {out.shape} vs {mask.shape}"
            out = tf.add(
                tf.multiply(out, mask),
                (1.0 - mask) * (-1e10),
                name="Q_mult_K_scaled_masked",
            )

        out = tf.nn.softmax(out)
        out = tf.matmul(out, V, name="attn_block_out")

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
        Q = tf.layers.dense(query, dim_model, activation=tf.nn.relu, name="Q")
        K = tf.layers.dense(memory, dim_model, activation=tf.nn.relu, name="K")
        V = tf.layers.dense(memory, dim_model, activation=tf.nn.relu, name="V")

        Q_split = tf.concat(
            tf.split(Q, nbr_heads, axis=2), axis=0, name="Q_multihead"
        )
        K_split = tf.concat(
            tf.split(K, nbr_heads, axis=2), axis=0, name="K_multihead"
        )
        V_split = tf.concat(
            tf.split(V, nbr_heads, axis=2), axis=0, name="V_multihead"
        )

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
        out = tf.concat(
            tf.split(out, nbr_heads, axis=0), axis=2, name="multihead_out"
        )

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
            + pointwise_feedforward(x=out, dim_ff=dim_ff, dim_model=dim_model),
            center=True,
            scale=True,
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
                scope="self_attn",
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
                scope="decorder_attn",
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

        if input_mask is not None:
            input_mask = input_mask[:, 0:1, :]

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
    """
    Creates a transformer optimized for Scene Memory. It expects the current
    observation to be a single element and not a sequence (as is normal) for
    transformers.
    """
    assert (
        observation.shape.ndims == memory.shape.ndims == 3
    ), "Incorrect tensor ranks."
    assert observation.shape[0] == memory.shape[0], "Mismatching batch sizes"
    assert (
        dim_model % nbr_heads == 0
    ), "dim_model must be divisible by nbr_heads"
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
