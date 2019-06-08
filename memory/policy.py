import warnings
from itertools import zip_longest
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from stable_baselines.a2c.utils import (
    conv,
    linear,
    conv_to_fc,
    batch_to_seq,
    seq_to_batch,
    lstm,
)
from stable_baselines.common.distributions import (
    make_proba_dist_type,
    CategoricalProbabilityDistribution,
    MultiCategoricalProbabilityDistribution,
    DiagGaussianProbabilityDistribution,
    BernoulliProbabilityDistribution,
)
from stable_baselines.common.input import observation_input
from stable_baselines.common.policies import RecurrentActorCriticPolicy, nature_cnn

from .memory import batch_update_memory
from .blocks import create_transformer


class SceneMemoryPolicy(RecurrentActorCriticPolicy):
    """
    SceneMemoryPolicy implements a policy that uses a Scene Memory Transformer
    to attend previous states as a memory.

    :param type sess: Description of parameter `sess`.
    :param type ob_space: Description of parameter `ob_space`.
    :param type ac_space: Description of parameter `ac_space`.
    :param type n_env: Description of parameter `n_env`.
    :param type n_steps: Description of parameter `n_steps`.
    :param type n_batch: Description of parameter `n_batch`.
    :param type memory_size: Description of parameter `memory_size`.
    :param type embedding_size: Description of parameter `embedding_size`.
    :param type transformer_ff_dim: Description of parameter `transformer_ff_dim`.
    :param type transformer_nbr_heads: Description of parameter `transformer_nbr_heads`.
    :param type transformer_nbr_encoders: Description of parameter `transformer_nbr_encoders`.
    :param type transformer_nbr_decoders: Description of parameter `transformer_nbr_decoders`.
    :param type extractor: Description of parameter `extractor`.
    :param type post_processor: Description of parameter `post_processor`.
    :param type reuse: Description of parameter `reuse`.
    :param type scale_features: Description of parameter `scale_features`.
    :param type **kwargs: Description of parameter `**kwargs`.

    """

    recurrent = True

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        memory_size=128,
        embedding_size=64,
        transformer_ff_dim=128,
        transformer_nbr_heads=8,
        transformer_nbr_encoders=6,
        transformer_nbr_decoders=6,
        extractor=None,
        post_processor=None,
        reuse=False,
        scale_features=False,
        **kwargs,
    ):
        super(SceneMemoryPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            state_shape=(memory_size, embedding_size + 1),
            reuse=reuse,
            scale=scale_features,
        )

        with tf.variable_scope("model", reuse=reuse):
            if extractor is not None:
                ext = extractor(self.processed_obs, **kwargs)
            else:
                ext = self.processed_obs

            extracted_features = linear(
                tf.keras.layers.Flatten()(ext), "extracted_features", embedding_size
            )

            assert extracted_features.shape[-1] == tf.Dimension(
                embedding_size
            ), f"embedding_size not correct: {extracted_features.shape[-1]} vs {embedding_size}"

            # Transform from (batch x seq, ... ) into (batch, seq, ...) shape
            sequence_input = tf.reshape(
                extracted_features,
                (self.n_env, n_steps, embedding_size),
                name="sequence_input",
            )
            sequence_state = tf.reshape(
                self.states_ph,
                (self.n_env, 1, memory_size, embedding_size + 1),
                name="sequence_state",
            )
            sequence_done = tf.reshape(
                self.dones_ph, (self.n_env, n_steps), name="sequence_done"
            )
            sequence_memory = tf.squeeze(
                sequence_state[:, :, :, :-1], axis=[1], name="sequence_memory"
            )
            sequence_mask = tf.squeeze(
                sequence_state[:, :, :, -1:], axis=[1, 3], name="sequence_mask"
            )

            # Update the memory states for all observations taking batches and
            # sequences into account.
            batch_memory, batch_mask, batch_new_state = batch_update_memory(
                observations=sequence_input,
                start_memory=sequence_memory,
                start_mask=sequence_mask,
                dones_ph=sequence_done,
            )

            # Transform back into (batch, ...) format
            memory = tf.reshape(batch_memory, (n_batch, memory_size, embedding_size))
            mask = tf.reshape(batch_mask, (n_batch, memory_size))
            new_state = tf.reshape(
                batch_new_state, (n_env, memory_size, embedding_size + 1)
            )
            self.snew = new_state

            # Mask should be of dims: (batch, memory_size, memory_size)
            tiled_mask = tf.tile(
                tf.reshape(mask, (n_batch, 1, memory_size)), (1, memory_size, 1)
            )

            # We need to tile the observation in the (transformer's) sequence
            # dimension. We do this since we use the current observation as the
            # context when attending each memory cell in the sequence.
            tiled_obs = tf.tile(
                tf.reshape(extracted_features, (n_batch, 1, embedding_size)),
                (1, memory_size, 1),
            )

            # Create the transformer.
            # Note that here the batch and seq has been turned into a single
            # dimension. This is due to that fact that we use sequence dimension
            # in the transformer to represent the memory dimension.
            trans_out = create_transformer(
                observation=tiled_obs,
                memory=memory,
                dim_model=embedding_size,
                dim_ff=transformer_ff_dim,
                nbr_heads=transformer_nbr_heads,
                nbr_encoders=transformer_nbr_encoders,
                nbr_decoders=transformer_nbr_decoders,
                input_mask=tiled_mask,
                target_mask=None,
            )
            flat_out = tf.keras.layers.Flatten()(trans_out)
            if post_processor is not None:
                flat_out = post_processor(flat_out, **kwargs)
            value_fn = linear(flat_out, "vf", 1)

            self._proba_distribution, self._policy, self.q_value = self.pdtype.proba_distribution_from_latent(
                flat_out, flat_out
            )

        self._value_fn = value_fn

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run(
                [self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask},
            )
        else:
            return self.sess.run(
                [self.action, self.value_flat, self.snew, self.neglogp],
                {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask},
            )

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(
            self.policy_proba,
            {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask},
        )

    def value(self, obs, state=None, mask=None):
        return self.sess.run(
            self.value_flat,
            {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask},
        )
