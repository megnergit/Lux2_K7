import luxai_s2
import numpy as np
import importlib
import importlib_metadata 
importlib.reload(importlib_metadata)

from pathlib import Path
Path('.').cwd()


import sys
# sys.path
# sys.path.insert(0, './v1/')
sys.path.insert(0, '../LuxDesign-S2/luxai_s2/')

from typing import Any, Dict, Callable
import numpy.typing as npt
from gym import spaces
import gym

import luxai_s2.env
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import ActionType, BidActionType, FactoryPlacementActionType
from luxai_s2.utils import my_turn_to_place_factory

from luxai_s2.wrappers.controllers import (
    Controller,
)
from pathlib import Path


# sys.path
# print(Path('.').cwd())
# import lux_v1
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
# from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from luxai_s2.wrappers import (
    SB3Wrapper,
)
# from lux_v1 import SimpleSingleUnitDiscreteController, SingleUnitObservationWrapper
#     
#    
#=============================================================

class Controller:
    def __init__(self, action_space: spaces.Space) -> None: 
        self.action_space = action_space

    def action_to_lux_action(
            self, agent: str, obs: Dict[str, Any], action: npt.NDArray):
        
        raise NotImplementedError()
    
    def action_masks(self, agent: str, obs: Dict[str, Any]):
        raise NotImplementedError()
    
class SimpleUnitDiscreteController(Controller):
    def __init__(self, env_cfg) -> None: 

        self.env_cfg = env_cfg
        self.move_act_dims = 4
        self.transfer_act_dims = 5
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims    
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.no_op_dim_high = self.dig_dim_high + self.no_op_dims

        self.total_act_dims = self.no_op_dim_high
        action_space = spaces.Discrete(self.total_act_dims)
        super().__init__(action_space)
    

    def _is_move_action(self, id):
        return id < self.move_dim_high
    
    def _get_move_action(self, id):
        return np.array([0, id+1, 0, 0, 0, 1])
    
    def _is_transfer_action(self, id):
        id = id - self.move_dim_high
        transfer_dir = id
        return np.array([1, transfer_dir, 0, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high
    
    def _get_pickup_action(self, id):

        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])    

    def _is_dig_action(self, id):
        return np.array([3, 0, 0, 0, 0, 1])

    def action_to_lux_action(
            self, agent:str, obs: Dict[str, Any], action: npt.NDArray):
        shared_obs = obs["player_0"]
        lux_action = dict()
        units = shared_obs["units"][agent]

        for unit_id in units.keys():
            unit = units[unit_id]
            choice = actionnaction_queue = []
            no_op = False

            if self._is_move_action(choice):
                action_queue = [self._get_move_action(choice)]
            elif self._is_transfer_action(choice):
                action_queue = [self._get_transfer_action(choice)]
            elif self._is_pickup_action(choice):
                action_queue = [self._get_pickup_action(choice)]
            elif self._is_dig_action(choice):
                action_queue = [self._get_dig_action(choice)]
            else: 

                no_op = True

            if len(unit["action_queue"]) >  0 and len(action_queue) > 0:
                same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                if same_actions:
                    no_op = True

            if not no_op: 
                lux_action[unit_id] = action_queue

            break

        factories = shared_obs["factories"][agent]
        if len(units) == 0:
            for unit_id in factories.keys():
                lux_action[unit_id] = 1

        return lux_action
    
    def action_makes(self, agent: str, obs: Dict[str, Any]):
        shared_obs = obs[agent]
        factory_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        factories = dict()

        for player in shared_obs["factories"]:
            factories[player] = dict()
            for unit_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][unit_id]
                f_pos = f_data["pos"]

                factory_occupancy_map[
                    f_pos[0] - 1 : f_pos[0] + 1, 
                    f_pos[1] - 1 : f_pos[1] + 2
                ] = f_data["strain_id"]

        units = shared_obs["units"][agent]
        action_mask = np.zeros((self.total_act_dims), dtype=bool)

        for unit_id in units.keys():
            action_mask = np.zeros(self.total_act_dims)

            action_mask[:4] = True

            unit = units[unit_id]

            pos = np.array(unit["pos"])

            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )
                if (
                    transfer_pos[0] < 0
                    or transfer_pos[1] < 0
                    or transfer_pos[0] >= len(factory_occupancy_map)
                    or transfer_pos[1] >= len(factory_occupancy_map[0])
                ):
                    continue

                factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]
                if factory_there in shared_obs["teams"][agent]["factory_strains"]:
                    action_mask[
                        self.transfer_dim_high - self.transfer_act_dims + i
                    ] = True
            factory_there = factory_occupancy_map[pos[0], pos[1]]
            on_top_of_factory = (
                factory_there in shared_obs["teams"][agent]["factory_strains"]
            )

            board_sum = (
                shared_obs["board"]["ice"][pos[0], pos[1]]
                + shared_obs["board"]["ore"][pos[0], pos[1]]
                + shared_obs["board"]["rubble"][pos[0], pos[1]]
                + shared_obs["board"]["lichen"][pos[0], pos[1]]
            )

            if board_sum > 0 and not on_top_of_factory : 
                action_mask[
                    self.pickup_dim_high - self.pickup_act_dims : self.pickup_dim_high
                ] = True

                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = False
                
            action_mask[-1] = True
            break
        return action_mask
    
#=============================================================

class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(13,))

    def observation(self, obs):
        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        for agent in obs.keys():
            obs_vec = np.zeros(
                13,
            )

            factories = shared_obs["factories"][agent]
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                break
            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]

                # store cargo+power values scaled to [0, 1]
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                cargo_vec = np.array(
                    [
                        unit["power"] / battery_cap,
                        unit["cargo"]["ice"] / cargo_space,
                        unit["cargo"]["ore"] / cargo_space,
                        unit["cargo"]["water"] / cargo_space,
                        unit["cargo"]["metal"] / cargo_space,
                    ]
                )
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light
                # normalize the unit position
                pos = np.array(unit["pos"]) / env_cfg.map_size
                unit_vec = np.concatenate(
                    [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )

                # we add some engineered features down here
                # compute closest ice tile
                ice_tile_distances = np.mean(
                    (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                # normalize the ice tile location
                closest_ice_tile = (
                    ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
                )
                obs_vec = np.concatenate(
                    [unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1
                )
                break
            observation[agent] = obs_vec

        return observation
    
#=============================================================
class SB3Wrapper(gym.Wrapper):
    def __init__(
        self, 
        env: LuxAI_S2,
        bid_policy: Callable[
            [str, ObservationStateDict], Dict[str, BidActionType]
        ]  = None,
        factory_placement_policy: Callable[
            [str, ObservationStateDict], Dict[str, FactoryPlacementActionType]
        ] = None,
        controller: Controller= None, 
    ) -> None: 
        
        gym.Wrapper.__init__(self, env)
        self.env = env

        assert controller is not None

        self.controller = controller
        self.action_space = controller.action_space

        if factory_placement_policy is None:

            def factory_placement_policy(player, obs: ObservationStateDict):
                potential_spawns = np.array(
                    list(zip(*np.where(obs["board"]["valid_spawns_mask"] ==1)))
                )
                spawn_loc = potential_spawns[
                    np.random.randint(0, len(potential_spawns))
                ]
                return dict(spawn=spawn_loc, metal=150, watner=150)
            
        self.factory_placement_policy = factory_placement_policy
        if bid_policy is None:
            def bid_policy(player, obs: ObservationStateDict):
                faction = "AlphaStrike"
                if player == "player_1": 
                    faction = "MotherMars"
                return dict(bid=0, faction=faction)
            
        self.bid_policy = bid_policy
        self.prev_obs = None

    def step(self, action: Dict[str, npt.NDArray]):

        lux_action = dict()
        for agent in self.env.agents:
            if agent in action:
                lux_action[agent] = self.controller.action_to_lux_action(
                    agent = agent, obs=self.prev_obs, action=action[agent]
                )
            else: 
                lux_action[agent] = dict()

        obs, reward, done, info = self.env.step(lux_action) 
        self.prev_obs = obs
        return obs, reward, done, info
    
    def reset(self, **kwargs):

        obs = self.env.reset(**kwargs)
        action=dict()
        for agent in self.env.agents:
            action[agent]=self.bid_policy(agent, obs[agent])
        obs, _, _, _ = self.env.step(action)

        while self.env.state.real_env_steps < 0:
            action = dict()
            for agent in self.env.agents:
                if my_turn_to_place_factory(
                    obs["player_0"]["teams"][agent]["place_first"],
                    self.env.state.env_steps,
                ):
                    action[agent]=self.factory_placement_policy(agent, obs[agent])
                else:
                    action[agent] = dict()
            obs, _, _, _, = self.env.step(action)
        self.prev_obs = obs

        return obs
    
#=============================================================
def zero_bid(player, obs):
    faction = "AlphaStrike"
    if player == "player_1":
        faction = "MotherMars"

    return dict(bid=0, faction=faction) 

#-------------------------------------------------------------
def place_near_random_ice(player, obs):

    if obs["teams"][player]["metal"] == 0:
        return dict()

    potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"]==1)))
    potential_spawns_set = set(potential_spawns)
    done_search = False

    ice_diff = np.diff(obs["board"]["ice"])
    pot_ice_spots = np.argwhere(ice_diff==1)

    if len(pot_ice_spots)==0:
        pot_ice_spots = potential_spawns

    trials = 5

    while trials > 0: 
        pos_idx = np.random.randint(0, len(pot_ice_spots))
        pos = pot_ice_spots[pos_idx]
        area = 3

        for x in range(area):
            for y in range(area):
                check_pos = [pos[0] + x - area / 2, pos[1] + y - area //2]
                if tuple(check_pos) in potential_spawns_set:
                    done_search = True
                    pos = check_pos
                    break
            if done_search:
                break
        if done_search:
            break
        trials -= 1

    if not done_search:
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        pos = spawn_loc

    metal = obs["teams"][player]["metal"]
    return dict(spawns=pos, metal=metal, water=metal)


#=============================================================

class CutomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None: 

        super().__init__(env)
        self.prevstep_metrics = None

    def step(self, action):
        agent = "player_0"
        opp_agent = "player_1"

        opp_factories = self.env.state.factories[opp_agent]

        for k in opp_factories.keys():
            factory = opp_factories[k]
            factory.cargo.water=1000

        action = {agent: action}
        obs, _, done, info = self.env.step(action)
        obs = obs[agent]
        done = done[agent]

        stats: StatsStateDict = self.env.state.stats[agent]

        info = dict()
        metrics = dict()
        metrics["ice_dug"] = (
            stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
        )
        metrics["water_produced"] = stats["generation"]["water"]

        metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
        metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

        info["metrics"] = metrics

        reward = 0

        if self.prev_step_metrics is not None:
            ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]

            water_produced_this_step = (
                metrics["water_produce"] - self.prev_step_metrics["water_produced"]
            )
            reward = ice_dug_this_step / 100 + water_produced_this_step

        self.prevstep_metrics = copy.deepcopy(metrics)
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)["player_0"]
        self.prev_step_metrics = None
        return obs
    

#=============================================================
# #=============================================================
# # how to reset environment
# #=============================================================
# env = gym.make("LuxAI_S2-v0")
# env.reset(seed=0)
# img = env.render("rgb_array")
# plt.imshow(img)

# #=============================================================

# env = gym.make("LuxAI_S2-v0")
# env = SB3Wrapper(env, zero_bid, place_near_random_ice, 
#                  controller=SimpleUnitDiscreteController(env.env_cfg))
# env.reset(seed=0)
# img = env.render("rgb_array")
# plt.imshow(img)

#=============================================================
# define environment
#=============================================================
def make_env(env_id: str, rank: int, seed: int = 0, max_episode_steps=200):
    def _init() -> gym.Env:

        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)

        env = SB3Wrapper(
            env, 
            controller=SimpleUnitDiscreteController(env.env_cfg),
            factory_placement_policy=place_near_random_ice,
        )

        env = SimpleUnitObservationWrapper(
            env
        )

        env = CutomEnvWrapper(env)

        env = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )

        env = Monitor(env)

        env.reset(seed=seed + rank)
        set_random_seed(seed)

        return env 
    
    return _init

#=============================================================

class TensorboardCallback(BaseCallback):
    def __init__(self, tag: str, verbose=0):
        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        c = 0

        for i, done in enumerate(self.locals["dones"]):
            if done: 
                info = self.locals["infos"][i]
                c += 1
                for k in info["metrics"]:
                    stat = info["metrics"][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True
                    
#=============================================================

#=============================================================
# END
#=============================================================
# scratch



#=============================================================