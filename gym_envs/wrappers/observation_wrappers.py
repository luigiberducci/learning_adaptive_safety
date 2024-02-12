from __future__ import annotations
from typing import Dict, Any

import gymnasium as gym
import numpy as np

from gym_envs.multi_agent_env.common.track import (
    extract_forward_curvature,
    extract_forward_raceline,
)
from gym_envs.multi_agent_env import MultiAgentRaceEnv
from gym_envs.multi_agent_env.common.utils import traj_global2local


def assert_vehicle_features(
    vehicle_features: list[str], agents_ids: list[str], obs_space: gym.Space
) -> None:
    """
    Assert that the vehicle features are in the observation space of the agents
    :param features: list of string, identifying the vehicle features
    :param agents_ids: list of string, identifying the agents
    :param obs_space: dictionary observation space
    """
    assert isinstance(vehicle_features, list), "vehicle_features must be a list"
    assert all(
        isinstance(feature, str) for feature in vehicle_features
    ), "vehicle_features must be a list of strings"
    assert isinstance(agents_ids, list), "agents_ids must be a list"
    assert all(
        isinstance(agent_id, str) for agent_id in agents_ids
    ), "agents_ids must be a list of strings"
    assert isinstance(obs_space, gym.spaces.Dict), "obs_space must be a gym.spaces.Dict"
    for agent_id in agents_ids:
        for feature in vehicle_features:
            assert agent_id in obs_space.spaces, f"{agent_id} not in obs-space"
            assert (
                feature in obs_space[agent_id].spaces
            ), f"{feature} not in obs-space of {agent_id}"


class VehicleTrackObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: MultiAgentRaceEnv,
        vehicle_features: list = None,
        track_features: list = None,
        forward_curv_lookahead: float = 10.0,
        n_curvature_points: int = 10,
        n_raceline_points: int = 10,
        n_obs_vehicles: int = 1,
    ):
        super().__init__(env)

        # observation configuration
        self.vehicle_features = (
            vehicle_features
            if vehicle_features is not None
            else ["pose", "velocity", "frenet_coords"]
        )
        self.track_features = (
            track_features if track_features is not None else ["curvature"]
        )
        self.forward_curv_lookahead = forward_curv_lookahead  # meters
        self.n_curvature_points = (
            n_curvature_points  # number of curvature points to sample
        )
        self.n_raceline_points = (
            n_raceline_points  # number of raceline points to sample
        )
        self.n_vehicles = 1 + n_obs_vehicles

        assert_vehicle_features(
            vehicle_features=self.vehicle_features,
            agents_ids=self.agents_ids,
            obs_space=self.observation_space,
        )

        # prepare new observation space
        # vehicle features
        obs_dict = {}
        for i in range(self.n_vehicles):
            agent_obs = {}
            for feature in self.vehicle_features:
                agent_obs[feature] = env.observation_space["ego"].spaces[feature]
            agent_id = f"vehicle_nearby{i}" if i > 0 else "ego"
            obs_dict[agent_id] = gym.spaces.Dict(agent_obs)
        # track features
        for feature in self.track_features:
            obs_dict[feature] = self.track_feature_space_factory(feature)

        self.observation_space = gym.spaces.Dict(obs_dict)

        # aux
        centerline = self.track.centerline
        self.centerline_xyk = np.stack(
            [
                centerline.xs,
                centerline.ys,
                centerline.kappas,
            ],
            axis=1,
        )
        raceline = self.track.raceline
        self.raceline_xyv = np.stack(
            [
                raceline.xs,
                raceline.ys,
                raceline.velxs,
            ],
            axis=1,
        )

    def track_feature_space_factory(self, feature_name: str) -> gym.spaces.Box:
        if feature_name == "curvature":
            return gym.spaces.Box(low=-10, high=10, shape=(1, self.n_curvature_points))
        if feature_name == "raceline":
            return gym.spaces.Box(low=-10, high=10, shape=(3, self.n_raceline_points))
        elif feature_name == "time_to_go":
            return gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        else:
            raise ValueError(f"unknown track feature {feature_name}")

    def observe_context_feature(
        self, feature_name: str, obs: Dict[str, Any]
    ) -> np.ndarray:
        if feature_name == "curvature":
            extract_forward_curvature(
                trajectory_xyk=self.centerline_xyk,
                point=obs["ego"]["pose"][:2],
                forward_distance=self.forward_curv_lookahead,
                n_points=self.n_curvature_points,
            )
        elif feature_name == "raceline":
            """
            extract forward raceline from the track centerline.
            the raceline profile is sampled at a fixed interval (5 wps) for a fixed length (10 samples).
            so in total, the raceline profile is 5*10 waypoints long (approx. 10 meters in General1).
            """
            ego_pose = obs["ego"]["pose"]

            waypoints = extract_forward_raceline(
                trajectory_xyv=self.raceline_xyv,
                point=ego_pose[:2],
                forward_distance=self.forward_curv_lookahead,
                n_points=self.n_raceline_points,
            )

            waypoints[:, :2] = traj_global2local(ego_pose, waypoints[:, :2])
            return waypoints
        elif feature_name == "time_to_go":
            """
            extract the remaining time to the end of the episode.
            """
            time_limit = self.env.env_params["termination"]["timeout"]
            return ((time_limit - obs["ego"]["time"]) / time_limit).astype(np.float32)
        else:
            raise ValueError(f"unknown track feature {feature_name}")

    def observation(self, observation):
        obs = {}
        ego_pose = observation["ego"]["pose"]
        all_poses = [observation[agent]["pose"] for agent in self.agents_ids]
        distances = np.linalg.norm(np.array(all_poses)[:, :2] - ego_pose[:2], axis=1)
        closest_ids = np.argsort(distances)[: self.n_vehicles]
        for i, nid in enumerate(closest_ids):
            orig_agent_id = self.agents_ids[nid]
            new_agent_id = "ego" if i == 0 else f"vehicle_nearby{i}"
            obs[new_agent_id] = {}
            for feature in self.vehicle_features:
                obs[new_agent_id][feature] = np.array(
                    observation[orig_agent_id][feature]
                )
        # zero padding for missing vehicles
        for i in range(self.n_vehicles - len(closest_ids)):
            new_agent_id = f"vehicle_nearby{len(closest_ids) + i}"
            obs[new_agent_id] = {}
            for feature in self.vehicle_features:
                obs[new_agent_id][feature] = np.zeros_like(
                    observation["ego"][feature]
                )
        for feature in self.track_features:
            obs[feature] = self.observe_context_feature(feature, observation)

        return obs


class VectorTrackObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        vehicle_features: list[str] = None,
        track_features: list[str] = None,
        forward_lookahead: float = 10.0,
        n_points: int = 10,
        n_observed_vehicles: int = 1,
    ):
        super().__init__(env)

        self.vehicle_features = vehicle_features or ["pose", "velocity"]
        self.track_features = track_features or ["left_boundary", "right_boundary"]

        self.forward_lookahead = forward_lookahead
        self.n_points = n_points
        self.n_observed_vehicles = n_observed_vehicles

        # check observation features are present in environment
        for agent_id in self.agents_ids:
            for feature in self.vehicle_features:
                assert (
                    feature in env.observation_space[agent_id].spaces
                ), f"{feature} not in obs-space of {agent_id}"

        # prepare new observation space
        # vehicle features
        obs_dict = {}
        # todo relative observation

        # track features
        for feature in self.track_features:
            obs_dict[feature] = self.track_feature_space_factory(feature)

        self.observation_space = gym.spaces.Dict(obs_dict)

    def track_feature_space_factory(self, feature_name: str) -> gym.spaces.Box:
        if feature_name == "left_boundary":
            return gym.spaces.Box(low=-10, high=10, shape=(self.n_points, 2))
        if feature_name == "right_boundary":
            return gym.spaces.Box(low=-10, high=10, shape=(self.n_points, 2))
        else:
            raise ValueError(f"unknown track feature {feature_name}")


if __name__ == "__main__":
    from gym_envs.multi_agent_env.planners.planner_factory import planner_factory
    from gym_envs.multi_agent_env.common.track import Track

    track = Track.from_track_name("General1")
    opp = planner_factory(planner="pp", track=track, agent_id="npc0")
    env = gym.make(
        "f110-multi-agent-v0",
        track_name="General1",
        npc_planners=[opp],
        render_mode="human",
    )

    env = VehicleTrackObservationWrapper(env)
    #env = VectorTrackObservationWrapper(env)

    print("action space:")
    print(env.action_space)
    print("observation space:")
    print(env.observation_space)

    obs, _ = env.reset(options={"mode": "random_back"})
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        env.render()
