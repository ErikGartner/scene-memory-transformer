import os

from gym.envs.classic_control import CartPoleEnv
from gym.wrappers.time_limit import TimeLimit
from gym import spaces
import gym
import numpy as np
import tensorflow as tf
import pytest

from stable_baselines import A2C, ACER, ACKTR, PPO2, bench
from stable_baselines.common.policies import (
    MlpLstmPolicy,
    LstmPolicy,
    nature_cnn,
)
from memory.policy import SceneMemoryPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.ppo2.ppo2 import safe_mean
from stable_baselines.a2c.utils import linear


def post_processor(inp, **kwargs):
    """Layers applied after the SMT, but before the softmax"""
    out = tf.nn.tanh(linear(inp, "post1", 64, init_scale=np.sqrt(2)))
    return out


class CustomSceneMemoryPolicyCartPole(SceneMemoryPolicy):
    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        memory_size=128,
        embedding_size=2,
        transformer_ff_dim=32,
        transformer_nbr_heads=1,
        transformer_nbr_encoders=3,
        transformer_nbr_decoders=3,
        reuse=False,
        **_kwargs
    ):
        super().__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            memory_size=memory_size,
            embedding_size=embedding_size,
            transformer_ff_dim=transformer_ff_dim,
            transformer_nbr_heads=transformer_nbr_heads,
            transformer_nbr_encoders=transformer_nbr_encoders,
            transformer_nbr_decoders=transformer_nbr_encoders,
            reuse=reuse,
            post_processor=post_processor,
            **_kwargs
        )


class CartPoleNoVelEnv(CartPoleEnv):
    """Variant of CartPoleEnv with velocity information removed. This task requires memory to solve."""

    def __init__(self):
        super(CartPoleNoVelEnv, self).__init__()
        high = np.array(
            [self.x_threshold * 2, self.theta_threshold_radians * 2]
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    @staticmethod
    def _pos_obs(full_obs):
        xpos, _xvel, thetapos, _thetavel = full_obs
        return xpos, thetapos

    def reset(self):
        full_obs = super().reset()
        return CartPoleNoVelEnv._pos_obs(full_obs)

    def step(self, action):
        full_obs, rew, done, info = super().step(action)
        return CartPoleNoVelEnv._pos_obs(full_obs), rew, done, info


NUM_ENVS = 16
NUM_EPISODES_FOR_SCORE = 10


def test_smt_train_cartpole():
    """Test that LSTM models are able to achieve >=150 (out of 500) reward on CartPoleNoVelEnv.
    This environment requires memory to perform well in."""

    def make_env(i):
        env = CartPoleNoVelEnv()
        env = bench.Monitor(env, None, allow_early_resets=True)
        env.seed(i)
        return env

    env = SubprocVecEnv([lambda: make_env(i) for i in range(NUM_ENVS)])
    env = VecNormalize(env)
    model = PPO2(
        CustomSceneMemoryPolicyCartPole,
        env,
        n_steps=128,
        nminibatches=NUM_ENVS,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        ent_coef=0.0,
        learning_rate=3e-4,
        cliprange=0.2,
        verbose=1,
        tensorboard_log="./logs/",
    )

    eprewmeans = []

    def reward_callback(local, _):
        nonlocal eprewmeans
        eprewmeans.append(
            safe_mean([ep_info["r"] for ep_info in local["ep_info_buf"]])
        )

    model.learn(total_timesteps=100000, seed=0, callback=reward_callback)

    # Maximum episode reward is 500.
    # In CartPole-v1, a non-recurrent policy can easily get >= 450.
    # In CartPoleNoVelEnv, a non-recurrent policy doesn't get more than ~50.
    # LSTM policies can reach above 400, but it varies a lot between runs; consistently get >=150.
    # See PR #244 for more detailed benchmarks.

    average_reward = (
        sum(eprewmeans[-NUM_EPISODES_FOR_SCORE:]) / NUM_EPISODES_FOR_SCORE
    )
    assert (
        average_reward >= 150
    ), "Mean reward below 150; per-episode rewards {}".format(average_reward)


@pytest.mark.skip(
    reason="this tests the official LSTM implementation as a reference"
)
def test_lstm_train_cartpole():
    """Test that LSTM models are able to achieve >=150 (out of 500) reward on CartPoleNoVelEnv.
    This environment requires memory to perform well in."""

    def make_env(i):
        env = CartPoleNoVelEnv()
        env = bench.Monitor(env, None, allow_early_resets=True)
        env.seed(i)
        return env

    env = SubprocVecEnv([lambda: make_env(i) for i in range(NUM_ENVS)])
    env = VecNormalize(env)
    model = PPO2(
        MlpLstmPolicy,
        env,
        n_steps=128,
        nminibatches=NUM_ENVS,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        ent_coef=0.0,
        learning_rate=3e-4,
        cliprange=0.2,
        verbose=1,
        tensorboard_log="./logs/",
    )

    eprewmeans = []

    def reward_callback(local, _):
        nonlocal eprewmeans
        eprewmeans.append(
            safe_mean([ep_info["r"] for ep_info in local["ep_info_buf"]])
        )

    model.learn(total_timesteps=100000, seed=0, callback=reward_callback)

    # Maximum episode reward is 500.
    # In CartPole-v1, a non-recurrent policy can easily get >= 450.
    # In CartPoleNoVelEnv, a non-recurrent policy doesn't get more than ~50.
    # LSTM policies can reach above 400, but it varies a lot between runs; consistently get >=150.
    # See PR #244 for more detailed benchmarks.

    average_reward = (
        sum(eprewmeans[-NUM_EPISODES_FOR_SCORE:]) / NUM_EPISODES_FOR_SCORE
    )
    assert (
        average_reward >= 150
    ), "Mean reward below 150; per-episode rewards {}".format(average_reward)
