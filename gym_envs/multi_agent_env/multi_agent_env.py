from typing import Dict, Any, List, Union

import gymnasium
import numpy as np

import pygame
from pydantic.utils import deep_update

from gym_envs.multi_agent_env.planners.planner import Planner
from gym_envs.multi_agent_env.common.utils import cartesian_to_frenet, nearest_point
from gym_envs.multi_agent_env.common.reset_fn import reset_fn_factory, RESET_MODES
from gym_envs.multi_agent_env.rendering import make_renderer
from gym_envs.multi_agent_env.rewards import reward_fn_factory
from gym_envs.multi_agent_env.common.termination_fn import termination_fn_factory
from gym_envs.multi_agent_env.common.track import Track
from f110_gym.envs import F110Env, DynamicsModel


def init_basic_ctrl(track, n_npcs):
    """return a basic pure-pursuit controller"""
    from gym_envs.multi_agent_env.planners.pure_pursuit import PurePursuitPlanner

    agent_ids = [f"npc{i}" for i in range(n_npcs)]
    return [PurePursuitPlanner(track, agent_id=agent_ids[i]) for i in range(n_npcs)]


class MultiAgentRaceEnv(gymnasium.Env):
    metadata = {
        "render_modes": ["human", "human_fast", "rgb_array"],
        "render_fps": 100,
    }

    def __init__(
            self,
            track_name: str,
            params: Dict[str, Any] = {},
            npc_planners: List[Planner] = [],
            seed: int = None,
            render_mode: str = "human",
    ):
        # params
        self._params = self._default_sim_params
        self._params = deep_update(self._params, params)
        self.metadata["render_fps"] = (
                100 // self._params["simulation"]["control_frequency"]
        )

        sim_params = self._params["simulation"]
        term_params = self._params["termination"]
        reset_params = self._params["reset"]
        reward_params = self._params["reward"]

        # track configuration
        self._track = Track.from_track_name(track_name)
        self.raceline_xyv = np.stack(
            [
                self._track.raceline.xs,
                self._track.raceline.ys,
                self.track.raceline.velxs,
            ],
            -1,
        )  # for frenet coords
        self.centerline_xyv = np.stack(
            [
                self._track.centerline.xs,
                self._track.centerline.ys,
                self.track.centerline.velxs,
            ],
            -1,
        )  # for frenet coords

        # multi-agent configuration
        self._n_npcs = len(npc_planners)
        self._npc_planners = npc_planners
        self.agents_ids = ["ego"] + [f"npc{i}" for i in range(self._n_npcs)]

        # seeding
        seed = np.random.randint(0, 1000000) if seed is None else seed

        # create simulation environment
        self.env = F110Env(
            map=self._track.filepath,
            map_ext=self._track.ext,
            params=sim_params["params"],
            dynamics_model=sim_params["dynamics_model"],
            num_agents=len(self.agents_ids),
            seed=seed,
        )

        # reset fns
        self.reset_fns = {
            reset_type: reset_fn_factory(self, reset_type)
            for reset_type in reset_params["types"]
        }

        # termination fns
        self.term_fns = {
            term_type: termination_fn_factory(term_type, **term_params)
            for term_type in term_params["types"]
        }

        # reward fn
        self.reward_fn = reward_fn_factory(self, **reward_params)

        self._scan_size = self.env.sim.agents[0].scan_simulator.num_beams
        self._scan_range = self.env.sim.agents[0].scan_simulator.max_range

        # keep state for playing npcs
        self._step = 0
        self._complete_state = None

        self.observation_space = gymnasium.spaces.Dict(
            {
                agent_id: gymnasium.spaces.Dict(
                    {
                        "scan": gymnasium.spaces.Box(
                            low=0.0, high=self._scan_range, shape=(self._scan_size,)
                        ),
                        "pose": gymnasium.spaces.Box(low=-1e5, high=1e5, shape=(3,)),
                        "frenet_coords": gymnasium.spaces.Box(
                            low=-1e5, high=1e5, shape=(2,)
                        ),
                        "velocity": gymnasium.spaces.Box(low=-5, high=20, shape=(1,)),
                        "collision": gymnasium.spaces.Box(
                            low=0.0, high=1.0, shape=(1,)
                        ),
                        "time": gymnasium.spaces.Box(low=0.0, high=1e5, shape=(1,)),
                        "nearest_waypoint_rl": gymnasium.spaces.Box(
                            low=-1e5, high=1e5, shape=(3,)
                        ),  # x, y, v
                    }
                )
                for agent_id in self.agents_ids
            }
        )

        steering_low, steering_high = (
            self.env.sim.params["s_min"],
            self.env.sim.params["s_max"],
        )
        velocity_low, velocity_high = 0.0, 10.0
        self.action_space = gymnasium.spaces.Dict(
            {
                "steering": gymnasium.spaces.Box(
                    low=steering_low, high=steering_high, shape=()
                ),
                "velocity": gymnasium.spaces.Box(
                    low=velocity_low, high=velocity_high, shape=()
                ),
            }
        )

        # render
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human_fast":
            self.metadata["render_fps"] *= 10  # boost fps by 10x
        self.renderer, self.render_spec = make_renderer(
            params=self.params,
            track=self.track,
            agent_ids=self.agents_ids,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
        )

    @property
    def state(self):
        return self._complete_state

    @state.setter
    def state(self, state):
        self._complete_state = state

    @property
    def track(self):
        return self._track

    @property
    def npc_planners(self):
        return self._npc_planners

    @property
    def env_params(self):
        return self._params

    @property
    def _default_sim_params(self):
        return {
            "name": "f110",
            "simulation": {
                "params": {
                    "mu": 1.0489,
                    "C_Sf": 4.718,
                    "C_Sr": 5.4562,
                    "lf": 0.15875,
                    "lr": 0.17145,
                    "h": 0.074,
                    "m": 3.74,
                    "I": 0.04712,
                    "s_min": -0.4189,
                    "s_max": 0.4189,
                    "sv_min": -3.2,
                    "sv_max": 3.2,
                    "v_switch": 7.319,
                    "a_max": 9.51,
                    "v_min": -5.0,
                    "v_max": 20.0,
                    "width": 0.31,
                    "length": 0.58,
                },
                "dynamics_model": DynamicsModel.KinematicBicycle_beta,
                "control_frequency": 1,  # 1 action every 1 simulation step
            },
            "observation": {
                "features": [
                    "scan",
                    "pose",
                    "frenet_coords",
                    "velocity",
                    "collision",
                    "time",
                    "nearest_waypoint_rl",
                ],
            },
            "action": {
                "type": "continuous",
            },
            "termination": {
                "types": ["on_collision", "on_all_complete_lap", "on_timeout"],
                "timeout": 100.0,
            },
            "reset": {
                "types": RESET_MODES,
                "default_reset_mode": "grid_random",
            },
            "reward": {
                "reward_type": "default",
            },
        }

    @staticmethod
    def _get_flat_action(
            action: Dict[str, Union[float, Dict[str, float]]]
    ) -> np.ndarray:
        check_single_agent = (
            lambda action: "steering" in action and "velocity" in action
        )
        check_multi_agent = lambda action: all(
            [check_single_agent(action[agent_id]) for agent_id in action]
        )
        assert check_single_agent(action) or check_multi_agent(
            action
        ), "Invalid action space"

        if "steering" in action and "velocity" in action:
            flat_action = [[action["steering"], action["velocity"]]]
        else:
            flat_action = []
            for agent_id in action.keys():
                flat_action.append(
                    [action[agent_id]["steering"], action[agent_id]["velocity"]]
                )

        flat_action = np.array(flat_action)
        assert len(flat_action.shape) == 2, "expected shape (n_agents, 2)"
        return flat_action

    def __getattr__(self, item):
        return getattr(self.env, item)

    def _extract_agent_state(self, obs, agent_id):
        assert agent_id in self.agents_ids
        n = self.agents_ids.index(agent_id)

        scan = np.clip(obs["scans"][n], 0, self._scan_range)
        pose = np.stack([obs["poses_x"][n], obs["poses_y"][n], obs["poses_theta"][n]])
        frenet_coords = cartesian_to_frenet(
            pose[:2], trajectory=self.centerline_xyv[:, :2]
        )
        velocity = obs["linear_vels_x"][n][None]
        collision = obs["collisions"][n][None]
        time = np.array([self.current_time])

        _, _, _, i = nearest_point(pose[:2], self.raceline_xyv[:, :2])
        nearest_waypoint_rl = self.raceline_xyv[i]

        progress = frenet_coords[0] - self._last_frenet_s[agent_id]
        if (
                abs(progress) > 1.0
        ):  # assumption: 1 meter of progress is due to crossing the finish line
            progress = 0.0

        tot_progress = self._tot_progress[agent_id] + progress
        self._tot_progress[agent_id] = tot_progress

        progress = np.array([progress], dtype=np.float32)
        tot_progress = np.array([tot_progress], dtype=np.float32)

        state = {
            "scan": scan,
            "pose": pose,
            "frenet_coords": frenet_coords,
            "velocity": velocity,
            "time": time,
            "collision": collision,
            "nearest_waypoint_rl": nearest_waypoint_rl,
            "progress": progress,
            "tot_progress": tot_progress,
        }

        return state

    def _prepare_state(self, old_obs):
        assert all(
            [
                f in old_obs
                for f in [
                "scans",
                "poses_x",
                "poses_y",
                "poses_theta",
                "linear_vels_x",
                "linear_vels_y",
                "ang_vels_z",
                "collisions",
            ]
            ]
        ), f"obs keys are {old_obs.keys()}"
        return {
            agent_id: self._extract_agent_state(old_obs, agent_id)
            for agent_id in self.agents_ids
        }

    def _prepare_obs(self, complete_state):
        features = self._params["observation"]["features"]
        obs = {
            agent_id: {
                f: complete_state[agent_id][f].astype(np.float32) for f in features
            }
            for agent_id in self.agents_ids
        }
        return obs

    def _prepare_info(self, old_obs, action, old_info):
        assert all(
            [
                f in old_obs
                for f in ["lap_times", "lap_counts", "collisions", "linear_vels_x"]
            ]
        ), f"obs keys are {old_obs.keys()}"
        assert all(
            [f in action for f in ["steering", "velocity"]]
        ), f"action keys are {action.keys()}"
        assert all(
            [f in old_info for f in ["checkpoint_done"]]
        ), f"info keys are {old_info.keys()}"
        info = {
            "lap_times": old_obs["lap_times"],
            "lap_counts": old_obs["lap_counts"],
            "collisions": old_obs["collisions"],
            "track_length": self.track.track_length,
            # next are never used
            # 'checkpoint_dones': old_info['checkpoint_done'],
            # 'velocities': old_obs['linear_vels_x'],
        }
        # store the state of all cars in the info dictionary (needed only for offline evaluation)
        # for constrained optimization
        cost = old_obs["collisions"][self.agents_ids.index("ego")]
        info.update(
            {
                "cost": cost,
                "cost_sum": cost,
            }
        )
        return info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed=seed)
        super().reset(seed=seed)

        # reset npc controllers
        ctrl_options = options if options is not None else {}
        [ctrl.reset(**ctrl_options) for ctrl in self._npc_planners]

        # reset terminal functions
        [term_fn.reset() for term_fn in self.term_fns.values()]

        # sample initial conditions according to the reset mode
        if options is not None:
            if "poses" in options:
                poses = options["poses"]
            elif "mode" in options:
                mode = options["mode"]
                assert any(
                    [reset in mode for reset in self.reset_fns.keys()]
                ), f"invalid reset mode {mode}"
                poses = self.reset_fns[mode].sample()
            else:
                mode = self._params["reset"]["default_reset_mode"]
                poses = self.reset_fns[mode].sample()
        else:
            mode = self._params["reset"]["default_reset_mode"]
            poses = self.reset_fns[mode].sample()

        # call original reset method
        original_obs, reward, done, original_info = self.env.reset(
            poses=np.array(poses)
        )

        # update observations and internal variables
        self._last_frenet_s = {
            agent: np.zeros(
                1,
            )
            for agent in self.agents_ids
        }
        self._tot_progress = {
            agent: np.zeros(
                1,
            )
            for agent in self.agents_ids
        }
        self.state = self._prepare_state(original_obs)
        dummy_action = {"steering": 0, "velocity": 0}
        obs = self._prepare_obs(self.state)
        info = self._prepare_info(
            original_obs, action=dummy_action, old_info=original_info
        )
        info.update({"state": self.state})
        self._step = 0
        self._last_frenet_s = {
            agent: self.state[agent]["frenet_coords"][0] for agent in self.agents_ids
        }

        # debug
        print(f"[debug - env {self._params['name']}] poses: {poses}")
        return obs, info

    def step(self, action: Union[np.ndarray, Dict[str, float]]):
        # `step` us called in the reset method of the original environment
        if type(action) == np.ndarray:
            return self.env.step(action)

        # here when `step` is called by the subclass environment
        # compute joint action of all agents (ie., ego + npcs)
        joint_action = self._prepare_multiagent_action(action)
        flatten_action = self._get_flat_action(joint_action)
        self.flatten_action = flatten_action  # for debugging

        # step the environment
        control_frequency = self._params["simulation"]["control_frequency"]
        reward = 0.0
        for _ in range(control_frequency):
            original_obs, r, done, original_info = self.env.step(flatten_action)
            reward += r
            if done:
                break

        # prepare the observation and info dictionaries
        complete_state = self._prepare_state(original_obs)
        next_obs = self._prepare_obs(complete_state)
        info = self._prepare_info(original_obs, action, original_info)

        # termination condition
        done = any(
            [
                term_fn(state=complete_state, info=info)
                for term_fn in self.term_fns.values()
            ]
        )
        truncated = False

        # compute the reward
        reward = self.reward_fn(self.state, action, complete_state, done)

        self._step += 1
        self.state = complete_state
        self._last_frenet_s = {
            agent: self.state[agent]["frenet_coords"][0] for agent in self.agents_ids
        }
        info.update({"state": self._complete_state})

        return next_obs, reward, done, truncated, info

    def _prepare_multiagent_action(self, action: Dict[str, float]):
        if len(self._npc_planners) == 0:
            return action

        joint_action = {"ego": action}
        for agent, controller in zip(self.agents_ids[1:], self._npc_planners):
            obs = self._complete_state
            joint_action[agent] = controller.plan(obs, agent_id=agent)

        return joint_action

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        self.renderer.add_renderer_callback(callback_func)
    def render(self):
        if self.render_mode not in self.metadata["render_modes"]:
            return

        self.renderer.update(state=self.render_obs)
        return self.renderer.render()

    def close(self):
        """
        Ensure renderer is closed upon deletion
        """
        if self.renderer is not None:
            self.renderer.close()
        super().close()


if __name__ == "__main__":
    track_name = "General1"
    track = Track.from_track_name(track_name)

    pps = init_basic_ctrl(track=track, n_npcs=2)

    env = gymnasium.make(
        "f110-multi-agent-v0",
        track_name=track_name,
        npc_planners=pps,
        params={
            "simulation": {
                "control_frequency": 1,
            },
            "reward": {"reward_type": "progress"},
            "reset": {"types": ["section0.0-0.5_random"]},
        },
        render_mode="human",
    )

    for i in range(10):
        print(f"episode {i + 1}")
        obs, _ = env.reset(options={"mode": "section0.0-0.5_random"})
        for j in range(500):
            action = env.action_space.sample()

            obs, reward, done, _, info = env.step(action)
            frame = env.render()

            print(info["state"]["ego"]["tot_progress"])
            if done:
                break

    # check env
    from gymnasium.wrappers import FlattenObservation
    from gym_envs.wrappers.action_wrappers import FlattenAction
    from gymnasium.utils.env_checker import check_env

    env = FlattenObservation(env)
    env = FlattenAction(env)

    try:
        check_env(env.unwrapped)
        print("[Result] env ok")
    except Exception as ex:
        print("[Result] env not compliant wt openai-gym standard")
        print(ex)
