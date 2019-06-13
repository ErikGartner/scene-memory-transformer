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


N_TRIALS = 100
MODELS = [A2C, PPO2]
LSTM_POLICIES = [CustomSceneMemoryPolicyCartPole]


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("policy", LSTM_POLICIES)
def test_scene_memory_policy(request, model_class, policy):
    model_fname = "./test_model_{}.pkl".format(request.node.name)

    try:
        # create and train
        if model_class == PPO2:
            model = model_class(policy, "CartPole-v1", nminibatches=1)
        else:
            model = model_class(policy, "CartPole-v1")
        model.learn(total_timesteps=100, seed=0)

        env = model.get_env()
        # predict and measure the acc reward
        obs = env.reset()
        for _ in range(N_TRIALS):
            action, _ = model.predict(obs)
            obs, _, _, _ = env.step(action)
        # saving
        model.save(model_fname)
        del model, env
        # loading
        _ = model_class.load(model_fname, policy=policy)

    finally:
        if os.path.exists(model_fname):
            os.remove(model_fname)
