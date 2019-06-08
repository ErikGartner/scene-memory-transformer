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
        n_lstm=256,
        reuse=False,
        layers=None,
        act_fun=tf.tanh,
        cnn_extractor=nature_cnn,
        layer_norm=False,
        feature_extraction="cnn",
        **kwargs,
    ):
        super(SceneMemoryPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            state_shape=(100, 64 + 1),
            reuse=reuse,
            scale=(feature_extraction == "cnn"),
        )

        self._kwargs_check(feature_extraction, kwargs)

        if layers is None:
            layers = [64, 64]
        else:
            warnings.warn(
                "The layers parameter is deprecated. Use the net_arch parameter instead."
            )

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                extracted_features = cnn_extractor(self.processed_obs, **kwargs)
            else:
                extracted_features = tf.layers.flatten(self.processed_obs)
                for i, layer_size in enumerate(layers):
                    extracted_features = act_fun(
                        linear(
                            extracted_features,
                            "pi_fc" + str(i),
                            n_hidden=layer_size,
                            init_scale=np.sqrt(2),
                        )
                    )

            print("Extracted features", extracted_features)
            input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
            masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)

            print(
                f"extracted_features: {extracted_features.shape}, input_sequence len: {len(input_sequence)}, {input_sequence[0].shape}, masks len: {len(masks)}, {masks[0].shape}"
            )

            self.embedding = extracted_features
            memory_size = 100
            embedding_size = extracted_features.shape.as_list()[-1]

            current_obs = extracted_features
            tiled_obs = tf.tile(
                tf.reshape(current_obs, (n_batch, 1, embedding_size)),
                (1, memory_size, 1),
            )
            print("tiled_obs", tiled_obs)

            # Create SMT module

            # self._input_mask_ph = tf.placeholder(
            #     tf.float32, name="input_mask", shape=(n_batch, memory_size)
            # )
            # input_mask_tiled = tf.tile(
            #     tf.expand_dims(self._input_mask_ph, 1), (1, memory_size, 1)
            # )
            # # self._target_mask_ph = tf.placeholder(
            # #     tf.float32,
            # #     name="target_mask",
            # #     shape=(n_batch, memory_size, memory_size),
            # # )
            # self._memory_ph = tf.placeholder(
            #     tf.float32, name="memory", shape=(n_batch, memory_size, embedding_size)
            # )
            print(
                f"states_ph: {self.states_ph.shape}, self.dones_ph: {self.dones_ph.shape}"
            )

            # # Split the states input into memory and input put mask
            # if self.states_ph.shape[0] == tf.Dimension(1):
            #     memory = tf.squeeze(self.states_ph[:, :, :-1], axis=[0])
            #     input_mask = tf.squeeze(self.states_ph[:, :, -1:], axis=[0, -1])
            #     batch_memory, batch_mask, new_state = batch_update_memory(
            #         observations=current_obs,
            #         start_memory=memory,
            #         start_mask=input_mask,
            #         dones_ph=self.dones_ph,
            #     )
            #     self.snew = new_state
            # else:
            #     # Multiple environments in parallell
            #     batch_memory = self.states_ph[:, :, :-1]
            #     batch_mask = tf.squeeze(self.states_ph[:, :, -1:], axis=[-1])
            #     self.snew = self.states_ph[:, :, :]

            # Transform into (batch, seq, ...) shape
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
            print(
                f"sequence_input: {sequence_input}, sequence_state: {sequence_state}, sequence_done: {sequence_done}, sequence_memory: {sequence_memory}, sequence_mask: {sequence_mask}"
            )

            batch_memory, batch_mask, batch_new_state = batch_update_memory(
                observations=sequence_input,
                start_memory=sequence_memory,
                start_mask=sequence_mask,
                dones_ph=sequence_done,
            )

            print(f"n_batch: {n_batch}")

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

            print(
                f"batch_memory: {batch_memory}, input_mask: {batch_mask}, self.snew: {self.snew}, tiled_mask: {tiled_mask}, tiled_obs: {tiled_obs}, memory: {memory}"
            )

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
        print("SETUP DONE!")

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
