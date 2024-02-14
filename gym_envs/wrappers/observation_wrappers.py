from typing import Dict, Any

import gymnasium as gym
import numpy as np

from gym_envs.multi_agent_env.common.track import (
    extract_forward_curvature,
    extract_forward_raceline,
)
from gym_envs.multi_agent_env import MultiAgentRaceEnv
from gym_envs.multi_agent_env.common.utils import traj_global2local


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

        # check observation features are present in environment
        for agent_id in self.agents_ids:
            for feature in self.vehicle_features:
                assert (
                    feature in env.observation_space[agent_id].spaces
                ), f"{feature} not in obs-space of {agent_id}"

        # prepare new observation space
        # vehicle features
        obs_dict = {}
        for i in range(self.n_vehicles):
            agent_obs = {}
            for feature in self.vehicle_features:
                agent_obs[feature] = env.observation_space["ego"].spaces[feature]
            obs_dict[f"vehicle_nearby{i}"] = gym.spaces.Dict(agent_obs)
        # track features
        for feature in self.track_features:
            obs_dict[feature] = self.track_feature_space_factory(feature)

        self.observation_space = gym.spaces.Dict(obs_dict)

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
            """
            extract forward curvature from the track centerline.
            the curvature profile is sampled at a fixed interval (5 wps) for a fixed length (10 samples).
            so in total, the curvature profile is 5*10 waypoints long (approx. 10 meters in General1).
            """
            ego_pos = obs["ego"]["pose"][:2]
            centerline_xyk = np.stack(
                [
                    self.track.centerline.xs,
                    self.track.centerline.ys,
                    self.track.centerline.kappas,
                ],
                axis=1,
            )

            ks = extract_forward_curvature(
                trajectory_xyk=centerline_xyk,
                point=ego_pos,
                forward_distance=self.forward_curv_lookahead,
                n_points=self.n_curvature_points,
            )
            ks = np.array(ks, dtype=np.float32)[None, :]
            return ks
        elif feature_name == "raceline":
            """
            extract forward raceline from the track centerline.
            the raceline profile is sampled at a fixed interval (5 wps) for a fixed length (10 samples).
            so in total, the raceline profile is 5*10 waypoints long (approx. 10 meters in General1).
            """
            ego_pose = obs["ego"]["pose"]
            raceline_xyv = np.stack(
                [
                    self.track.raceline.xs,
                    self.track.raceline.ys,
                    self.track.raceline.velxs,
                ],
                axis=1,
            )

            waypoints = extract_forward_raceline(
                trajectory_xyv=raceline_xyv,
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
            agent_id = self.agents_ids[nid]
            obs[f"vehicle_nearby{i}"] = {}
            for feature in self.vehicle_features:
                obs[f"vehicle_nearby{i}"][feature] = np.array(
                    observation[agent_id][feature]
                )
        # zero padding for missing vehicles
        for i in range(self.n_vehicles - len(closest_ids)):
            obs[f"vehicle_nearby{len(closest_ids) + i}"] = {}
            for feature in self.vehicle_features:
                obs[f"vehicle_nearby{len(closest_ids) + i}"][feature] = np.zeros_like(
                    observation["ego"][feature]
                )
        for feature in self.track_features:
            obs[feature] = self.observe_context_feature(feature, observation)

        return obs


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

    # rendering
    def render_waypoints(e):
        """
        Callback to render waypoints.
        """
        points = np.stack([track.raceline.xs, track.raceline.ys], axis=1)
        e.render_closed_lines(points, color=(128, 0, 0), size=1)

    env.add_render_callback(render_waypoints)

    env = VehicleTrackObservationWrapper(
        env,
        vehicle_features=["pose", "frenet_coords", "velocity"],
        track_features=["curvature", "raceline"],
    )

    print("action space:")
    print(env.action_space)
    print("observation space:")
    print(env.observation_space)

    obs, _ = env.reset(options={"mode": "random_back"})
    done = False
    debug = False

    t = 0
    while not done:
        t += 1
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        env.render()
