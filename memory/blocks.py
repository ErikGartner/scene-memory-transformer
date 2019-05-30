import tensorflow as tf
import tensorflow.contrib as tc


def scaled_dot_product_attention(
    Q: tf.tensor, K: tf.tensor, V: tf.tensor, scope: str = "sdp_attention"
) -> tf.tensor:
    """
    Scaled dot product attention.
    """

    assert Q.shape[-1] == K.shape[-1] == V.shape[-1]

    d = float(Q.shape[-1])

    with tf.variable_scope(scope):
        # Create K^T
        out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))

        # Scale by dimension
        out = out / tf.sqrt(d)

        out = tf.nn.softmax(out)
        out = tf.matmul(out, V)

    return out


def multihead_attention(
    query: tf.tensor,
    memory: tf.tensor,
    nbr_heads: int,
    dim_model: int,
    scope: str = "multihead_attention",
) -> tf.tensor:

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

        # Apply scaled dot product attention
        out = scaled_dot_product_attention(Q_split, K_split, V_split)

        # Merge the multi-head back to the original shape
        out = tf.concat(tf.split(out, nbr_heads, axis=0), axis=2)

    return out


def pointwise_feedforward(
    x: tf.tensor, dim_ff: int, dim_model: int, scope: str = "pointwise_feedforward"
) -> tf.tensor:

    out = x
    with tf.variable_scope(scope):
        out = tf.layers.conv1d(
            out, filters=dim_ff, kernel_size=1, activation=tf.nn.relu
        )
        out = tf.layers.conv1d(out, filters=dim_model, kernel_size=1)

    return out


def encoder_layer(
    x: tf.tensor, nbr_heads: int, dim_model: int, dim_ff: int, scope: str
) -> tf.tensor:

    out = x
    with tf.variable_scope(scope):
        out = tc.layers.layer_norm(
            out + multihead_attention(out, out, nbr_heads, dim_model),
            center=True,
            scale=True,
        )
        out = tc.layers.layer_norm(out + pointwise_feedforward(out, dim_ff, dim_model))

    return out
