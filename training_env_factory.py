import pathlib
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation, FrameStack

from gym_envs import cbf_factory
from gym_envs.multi_agent_env.common.track import Track
from gym_envs.multi_agent_env.planners.planner_factory import planner_factory
from gym_envs.wrappers import FrameSkip, FlattenAction, CBFSafetyLayer
from gym_envs.wrappers.action_wrappers import (
    LocalPathActionWrapper,
    WaypointActionWrapper,
)
from gym_envs.wrappers.observation_wrappers import VehicleTrackObservationWrapper


def make_f110_base_env(
    env_id,
    env_params,
    cbf_params,
    idx,
    capture_video,
    default_render_mode=None,
):
    track_name = env_params["track_name"]
    opp_planner = env_params["opp_planner"]
    opp_params = env_params["opp_params"]
    termination_types = env_params["termination_types"]
    timeout = env_params["timeout"]
    reward = env_params["reward"]
    reset_mode = env_params["reset_mode"]
    control_freq = env_params["control_freq"]
    planning_freq = env_params["planning_freq"]

    assert planning_freq % control_freq == 0, (
        "planning_freq must be divisible by control_freq,"
        f"got planning_freq={planning_freq} and control_freq={control_freq}"
    )

    if opp_planner is not None:
        track = Track.from_track_name(track_name)
        npc_planners = []
        for i, (opp_plan, opp_par) in enumerate(zip(opp_planner, opp_params)):
            opp = planner_factory(
                planner=opp_plan, track=track, params=opp_par, agent_id=f"npc{i}"
            )
            npc_planners.append(opp)
    else:
        npc_planners = []

    gym_env_params = {
        "track_name": track_name,
        "npc_planners": npc_planners,
        "params": {
            "name": idx,
            "simulation": {
                "control_frequency": control_freq,
            },
            "termination": {
                "types": termination_types,
                "timeout": timeout,
            },
            "reward": {
                "reward_type": reward,
            },
            "reset": {
                "types": [reset_mode],
                "default_reset_mode": reset_mode,
            },
        },
    }
    if capture_video:
        env = gym.make(env_id, **gym_env_params, render_mode="rgb_array")
    else:
        env = gym.make(env_id, **gym_env_params, render_mode=default_render_mode)

    env = FlattenAction(env=env)

    # cbf wrapper
    if cbf_params["use_cbf"]:
        make_cbf = cbf_factory(env_id=env_id, cbf_type=cbf_params["cbf_type"])
        env = CBFSafetyLayer(
            env,
            safety_dim=cbf_params["safety_dim"],
            gamma_range=cbf_params["cbf_gamma_range"],
            make_cbf=make_cbf,
        )

    return env


def make_f110_env(
    env_id: str,
    env_params: dict,
    cbf_params: dict,
    idx: int,
    evaluation: bool,
    capture_video: bool,
    log_dir: pathlib.Path,
    default_render_mode: str = None,
):
    def thunk():
        env = make_f110_base_env(
            env_id=env_id,
            idx=idx,
            env_params=env_params,
            cbf_params=cbf_params,
            capture_video=capture_video,
            default_render_mode=default_render_mode,
        )

        # planning-action wrapper
        control_freq = env_params["control_freq"]
        planning_freq = env_params["planning_freq"]
        frame_skip = planning_freq // control_freq
        if (
            env_params["local_path_generation"]
            in LocalPathActionWrapper.planner_fns.keys()
        ):
            env = LocalPathActionWrapper(
                env=env, planner_type=env_params["local_path_generation"]
            )
        else:
            env = WaypointActionWrapper(env=env)
        env = FrameSkip(env, skip=frame_skip)

        # observations
        vehicle_features = env_params["vehicle_features"]
        track_features = env_params["track_features"]
        lookahead = env_params["lookahead"]
        n_points = env_params["n_points"]

        env = VehicleTrackObservationWrapper(
            env,
            vehicle_features=vehicle_features,
            track_features=track_features,
            lookahead=lookahead,
            n_points=n_points,
        )
        env = FlattenObservation(env)
        env = FrameStack(env, cbf_params["frame_stack"])
        env = FlattenObservation(env)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if evaluation and capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, f"{log_dir}/videos", episode_trigger=lambda x: True
                )

        env = gym.wrappers.ClipAction(env)

        if not evaluation:
            env = gym.wrappers.NormalizeReward(env, gamma=0.99)
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10)
            )

        return env

    return thunk


def load_f110_env_params(parser_args) -> dict:
    params = {
        "track_name": parser_args.track_name,
        "opp_planner": parser_args.opp_planner,
        "opp_params": [
            {"vgain": mu, "vgain_std": std}
            for mu, std in zip(parser_args.opp_vgain, parser_args.opp_vgain_std)
        ],
        "termination_types": parser_args.termination_types,
        "timeout": parser_args.time_limit,
        "reset_mode": parser_args.reset_mode,
        "reward": parser_args.reward,
        "control_freq": parser_args.control_freq,
        "planning_freq": parser_args.planning_freq,
        "local_path_generation": parser_args.local_path_generation,
        "vehicle_features": parser_args.vehicle_features,
        "track_features": parser_args.track_features,
        "forward_curv_lookahead": parser_args.lookahead,
        "n_curvature_points": parser_args.n_points,
    }
    return params


def load_f110_cbf_params(parser_args):
    cbf_type = parser_args.cbf_type
    cbf_type = f"{cbf_type}decay" if parser_args.use_decay else cbf_type
    params = {
        "use_cbf": parser_args.use_cbf,
        "cbf_type": cbf_type,
        "safety_dim": parser_args.safety_dim,
        "cbf_gamma_range": parser_args.cbf_gamma_range,
        "frame_stack": parser_args.frame_stack,
    }
    return params


def make_particle_env(
    env_id,
    env_params,
    cbf_params,
    idx,
    evaluation,
    capture_video,
    log_dir,
    default_render_mode=None,
):
    def thunk():
        if capture_video:
            env = gym.make(env_id, params=env_params, render_mode="rgb_array")
        else:
            env = gym.make(env_id, params=env_params, render_mode=default_render_mode)

        env = FlattenObservation(env)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = FrameSkip(env, cbf_params["planning_freq"])
        env = FrameStack(env, cbf_params["frame_stack"])
        env = FlattenObservation(env)

        env = gym.wrappers.ClipAction(env)

        return env

    return thunk


def load_particle_env_params(parser_args) -> dict:
    params = {
        "world_size": parser_args.world_size,
        "time_limit": parser_args.time_limit,
        "min_dist_goal": parser_args.min_dist_goal,
        "min_agents": parser_args.min_agents,
        "max_agents": parser_args.max_agents,
        "obs_type": {"type": parser_args.obs_type},
        "cbf_gamma_range": parser_args.cbf_gamma_range,
        "ctrl_params": {
            "use_clf": parser_args.use_clf,
            "use_cbf": parser_args.use_cbf,
            "robust": bool(parser_args.cbf_type == "robust"),
            "safety_dim": parser_args.safety_dim,
        },
    }
    return params


def load_particle_cbf_params(parser_args) -> dict:
    return {
        "planning_freq": parser_args.planning_freq,
        "frame_stack": parser_args.frame_stack,
    }


def get_env_id(env_id: str, use_cbf: bool, use_ctrl: bool, use_decay: bool = False):
    if env_id == "particle-env-v0" or env_id == "particle-env-v1":
        env_v = env_id.split("-")[-1]
        ctrl = "auto" if use_ctrl else "rl"
        cbf = "simple" if use_cbf else "none"
        cbf = f"{cbf}decay" if use_cbf and use_decay else cbf
        env_id = f"particle-env-{ctrl}-{cbf}-{env_v}"
    elif env_id == "f110-multi-agent-v0" or env_id == "f110-multi-agent-v1":
        env_v = env_id.split("-")[-1]
        ctrl = "auto" if use_ctrl else "rl"
        cbf = "cbf" if use_cbf else "none"
        cbf = f"{cbf}decay" if use_cbf and use_decay else cbf
        env_id = f"f110-multi-agent-{ctrl}-{cbf}-{env_v}"
    else:
        raise ValueError(f"env_id {env_id} is not supported")

    return env_id


def load_env_param_factory(env_id: str) -> Callable:
    if env_id == "particle-env-v0":
        return load_particle_env_params
    elif env_id == "f110-multi-agent-v0":
        return load_f110_env_params
    else:
        raise NotImplementedError


def load_cbf_param_factory(env_id: str) -> Callable:
    if env_id == "particle-env-v0":
        return load_particle_cbf_params
    elif env_id == "f110-multi-agent-v0":
        return load_f110_cbf_params
    else:
        raise NotImplementedError


def make_env_factory(env_id: str) -> Callable:
    if env_id == "particle-env-v0":
        return make_particle_env
    elif env_id == "f110-multi-agent-v0":
        return make_f110_env
    else:
        raise NotImplementedError
