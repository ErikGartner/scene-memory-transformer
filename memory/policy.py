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
    Actor critic policy object uses a previous state in the computation for the current step.
    NOTE: this class is not limited to recurrent neural network policies,
    see https://github.com/hill-a/stable-baselines/issues/241
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param state_shape: (tuple<int>) shape of the per-environment state space.
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
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
        memory_size=100,
        embedding_size=64,
        extractor=None,
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
                dim_ff=50,
                nbr_heads=2,
                nbr_encoders=1,
                nbr_decoders=1,
                input_mask=tiled_mask,
                target_mask=None,
            )
            flat_out = tf.keras.layers.Flatten()(trans_out)
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
