# import luxai_s2
import numpy as np
import importlib
import importlib_metadata 
importlib.reload(importlib_metadata)

import sys
# sys.path.clear()
# sys.path.insert(0, './v1/')
# sys.path.insert(0, './')
# sys.path.insert(0, './Lux-Design-S2/luxai_s2/')
# sys.path.insert(0, '../Lux-Design-S2/luxai_s2/')

from lux_v1 import *
from typing import Any, Dict, Callable
import numpy.typing as npt
from gym import spaces
import gym

# from sb3 import SB3Wrapper
# from luxai_s2.wrappers.sb3 import SB3Wrapper
import luxai_s2.env
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import ActionType, BidActionType, FactoryPlacementActionType
from luxai_s2.utils import my_turn_to_place_factory

from luxai_s2.wrappers.controllers import Controller
from luxai_s2.wrappers import sb3

import matplotlib.pyplot as plt
import copy

from stable_baselines3.common.vec_env import SubprocVecEnv 
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import TimeLimit

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback 

import os.path as osp
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import PPO

from luxai_s2.state import ObservationStateDict, StatsStateDict, create_empty_stats
from luxai_s2.utils.heuristics.factory import build_single_heavy
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from luxai_s2.wrappers import (
     SB3Wrapper,
#     SimpleSingleUnitDiscreteController,
#     SingleUnitObservationWrapper,
)
import pretty_errors
import pdb
#=============================================================
# how to reset environment
#=============================================================
# env = gym.make("Lux
# AI_S2-v0")
# env.reset(seed=0)
# img = env.render("rgb_array")
# plt.imshow(img)

#=============================================================

# env = gym.make("LuxAI_S2-v0")
# env = SB3Wrapper(env, bid_policy=zero_bid, factory_placement_policy=place_near_random_ice, 
#                  controller=SimpleUnitDiscreteController(env.env_cfg))

# env.reset(seed=0)
# img = env.render("rgb_array")
# plt.imshow(img)
# # plt.show()

#=============================================================
# training setup
#=============================================================
if __name__ == "__main__":

    set_random_seed(42)
    log_path= "logs/exp_2"
    num_envs = 4

    # help(SubprocVecEnv)

    env = SubprocVecEnv(
        [make_env("LuxAI_S2-v0", i, max_episode_steps=200) for i in range(num_envs)]
        )

    env.reset()


    eval_env = SubprocVecEnv(
        [make_env("LuxAI_S2-v0", i, max_episode_steps=1000) for i in range(num_envs)]
        )


    eval_env.reset()

    rollout_steps = 4000
    policy_kwargs = dict(net_arch=(128,128))


    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=rollout_steps // num_envs, 
        batch_size=800,
        learning_rate=3e-4, 
        policy_kwargs=policy_kwargs,
        verbose=1, 
        n_epochs=2, 
        target_kl=0.05,
        gamma=0.99,
        tensorboard_log=osp.join(log_path),
    )
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=osp.join(log_path, "models"),
        log_path=osp.join(log_path, "eval_logs"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )

    #=============================================================
    # training
    #=============================================================

#     total_timesteps = 30_000_000
    total_timesteps = 30_000_000
    model.learn(
        total_timesteps, 
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
    )
    model.save(osp.join(log_path, "models/latest_model"))

#=============================================================
# END
#=============================================================

