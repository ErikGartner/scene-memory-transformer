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

from .memory import Memory
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
        **kwargs
    ):
        super(SceneMemoryPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            state_shape=(2 * n_lstm,),
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
            # input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
            # print("input", input_sequence)

            # print("input", self.states_ph)
            #    exit(0)
            # masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
            # rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
            #                             layer_norm=layer_norm)
            # rnn_output = seq_to_batch(rnn_output)

            memory_size = 100
            embedding_size = extracted_features.shape.as_list()[-1]

            current_obs = extracted_features
            current_obs = tf.tile(
                tf.reshape(current_obs, (n_batch, 1, embedding_size)),
                (1, memory_size, 1),
            )
            print("current_obs", current_obs)

            # Create SMT module
            self._memory = Memory(
                memory_size=memory_size, embedding_size=embedding_size
            )
            self._input_mask = tf.placeholder(
                tf.float32, name="input_mask", shape=(n_batch, memory_size, memory_size)
            )
            self._target_mask = tf.placeholder(
                tf.float32,
                name="target_mask",
                shape=(n_batch, memory_size, memory_size),
            )
            self._memory_ph = tf.placeholder(
                tf.float32,
                name="target_mask",
                shape=(n_batch, memory_size, embedding_size),
            )

            trans_out = create_transformer(
                observation=current_obs,
                memory=self._memory_ph,
                dim_model=embedding_size,
                dim_ff=50,
                nbr_heads=2,
                nbr_encoders=1,
                nbr_decoders=1,
                input_mask=self._input_mask,
                target_mask=self._target_mask,
            )

            print("trans_out", trans_out)
            flat_out = tf.keras.layers.Flatten()(trans_out)
            print("flat_out", flat_out)
            value_fn = linear(flat_out, "vf", 1)
            print("Value fn", value_fn)

            self._proba_distribution, self._policy, self.q_value = self.pdtype.proba_distribution_from_latent(
                flat_out, flat_out
            )

        self._value_fn = value_fn

        self._setup_init()
        print("SETUP DONE!")

    def step(self, obs, state=None, mask=None, deterministic=False):
        print("STEP", obs, state, mask, deterministic)
        if deterministic:
            return self.sess.run(
                [self.deterministic_action, self.value_flat, self.neglogp],
                {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask},
            )
        else:
            return self.sess.run(
                [self.action, self.value_flat, self.neglogp],
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
