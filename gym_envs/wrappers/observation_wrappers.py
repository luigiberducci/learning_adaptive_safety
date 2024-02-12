from __future__ import annotations
from typing import Dict, Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from gym_envs.multi_agent_env.common.track import (
    extract_forward_curvature,
    extract_forward_raceline,
    extract_forward_boundary,
    Raceline,
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
        lookahead: float = 10.0,
        n_points: int = 10,
        n_obs_vehicles: int = 1,
        relative_obs: bool = True,
    ):
        super().__init__(env)

        # observation configuration
        self.vehicle_features = vehicle_features or [
            "pose",
            "velocity",
            "frenet_coords",
        ]
        self.track_features = track_features or ["raceline"]

        self.lookahead = lookahead  # meters
        self.n_points = n_points
        self.n_vehicles = 1 + n_obs_vehicles
        self.relative_obs = relative_obs

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

        xys = np.stack([centerline.xs, centerline.ys], axis=1)
        xys = np.concatenate([xys, [xys[0]]], axis=0)
        dxys = xys[1:] - xys[:-1]
        normals = np.stack([-dxys[:, 1], dxys[:, 0]], axis=1)
        normals /= np.linalg.norm(normals, axis=1)[:, None]

        dlefts = centerline.dlefts
        drights = centerline.drights

        self.left_boundary_xy = xys[:-1] + dlefts[:, None] * normals
        self.right_boundary_xy = xys[:-1] - drights[:, None] * normals

    def track_feature_space_factory(self, feature_name: str) -> gym.spaces.Box:
        if feature_name == "curvature":
            return gym.spaces.Box(low=-10, high=10, shape=(1, self.n_points))
        elif feature_name == "raceline":
            return gym.spaces.Box(low=-10, high=10, shape=(3, self.n_points))
        elif feature_name == "time_to_go":
            return gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        elif feature_name == "left_boundary":
            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_points, 2),
                dtype=np.float32,
            )
        elif feature_name == "right_boundary":
            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_points, 2),
                dtype=np.float32,
            )
        elif feature_name == "vec_raceline":
            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_points, 3),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"unknown track feature {feature_name}")

    def observe_context_feature(
        self, feature_name: str, obs: Dict[str, Any]
    ) -> np.ndarray:
        if feature_name == "curvature":
            extract_forward_curvature(
                trajectory_xyk=self.centerline_xyk,
                point=obs["ego"]["pose"][:2],
                forward_distance=self.lookahead,
                n_points=self.n_points,
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
                forward_distance=self.lookahead,
                n_points=self.n_points,
            )

            waypoints[:, :2] = traj_global2local(ego_pose, waypoints[:, :2])
            return waypoints
        elif feature_name == "time_to_go":
            """
            extract the remaining time to the end of the episode.
            """
            time_limit = self.env.env_params["termination"]["timeout"]
            return ((time_limit - obs["ego"]["time"]) / time_limit).astype(np.float32)
        elif feature_name == "vec_raceline":
            """
            extract forward raceline from the track centerline.
            the raceline profile is sampled at a fixed interval (5 wps) for a fixed length (10 samples).
            so in total, the raceline profile is 5*10 waypoints long (approx. 10 meters in General1).
            """
            ego_pose = obs["ego"]["pose"]
            ego_vel = obs["ego"]["velocity"]
            ego_xyv = np.concatenate([ego_pose[:2], ego_vel])

            waypoints = extract_forward_raceline(
                trajectory_xyv=self.raceline_xyv,
                point=ego_xyv[:2],
                forward_distance=self.lookahead,
                n_points=self.n_points,
            )

            # vectorize the waypoints first -> vector between ego pose and first waypoint,
            # then vector between first and second waypoint, and so on.
            rel_waypoints = traj_global2local(ego_pose, waypoints[:, :2])
            rel_waypoints[1:] -= rel_waypoints[:-1]
            rel_waypoints = np.concatenate([rel_waypoints, waypoints[:, 2:]], axis=1)
            rel_waypoints[:, 2] -= ego_vel  # relative velocity

            return rel_waypoints
        elif feature_name in ["left_boundary", "right_boundary"]:
            ego_pose = obs["ego"]["pose"]

            raceline_wp = extract_forward_raceline(
                trajectory_xyv=self.raceline_xyv,
                point=ego_pose[:2],
                forward_distance=self.lookahead,
                n_points=self.n_points,
            )

            first, last = raceline_wp[0, :2], raceline_wp[-1, :2]

            all_boundary = (
                self.left_boundary_xy
                if feature_name == "left_boundary"
                else self.right_boundary_xy
            )
            boundary = extract_forward_boundary(
                trajectory_xy=all_boundary,
                first_point=first,
                last_point=last,
                n_points=self.n_points,
            )

            # vectorize the boundary
            rel_boundary = traj_global2local(ego_pose, boundary)
            rel_boundary[1:] -= rel_boundary[:-1]

            return rel_boundary
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
                ego_feature = np.array(observation["ego"][feature])
                other_feature = np.array(observation[orig_agent_id][feature])
                if orig_agent_id != "ego" and self.relative_obs:
                    if feature == "pose":
                        # transform pose to ego frame
                        rel_pos = traj_global2local(
                            ego_pose=ego_feature, traj=other_feature[:2][None]
                        )
                        rel_head = other_feature[2] - ego_feature[2]
                        proc_feature = np.array(
                            [rel_pos[0][0], rel_pos[0][1], rel_head]
                        )
                    else:
                        proc_feature = other_feature - ego_feature
                else:
                    proc_feature = other_feature
                obs[new_agent_id][feature] = proc_feature
        # zero padding for missing vehicles
        for i in range(self.n_vehicles - len(closest_ids)):
            new_agent_id = f"vehicle_nearby{len(closest_ids) + i}"
            obs[new_agent_id] = {}
            for feature in self.vehicle_features:
                obs[new_agent_id][feature] = np.zeros_like(observation["ego"][feature])
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
        track_features=["vec_raceline", "raceline", "left_boundary", "right_boundary"],
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

        # debug
        raceline = obs["raceline"]
        vec_raceline = obs["vec_raceline"]
        vec_lboundary = obs["left_boundary"]
        vec_rboundary = obs["right_boundary"]

        # plot arrows from 0,0 to rl points
        if debug and t % 50 == 0:
            fig, axes = plt.subplots(1, 2)
            ax0, ax1 = axes

            ax0.clear()
            for i in range(raceline.shape[0]):
                ax0.arrow(0, 0, raceline[i, 0], raceline[i, 1], head_width=0.02)

            # plot arrows from 0,0 to first rl point, then from first to second, etc.
            ax1.clear()
            for vec_polyline in [vec_raceline, vec_lboundary, vec_rboundary]:
                starting_pt = np.array([0.0, 0.0])
                for i in range(vec_polyline.shape[0] - 1):
                    starting_pt += vec_polyline[i][:2]
                    ax1.arrow(
                        starting_pt[0],
                        starting_pt[1],
                        vec_polyline[i + 1, 0],
                        vec_polyline[i + 1, 1],
                        head_width=0.02,
                    )

            # plot opp
            for agent_id, agent_obs in obs.items():
                if agent_id in env.track_features:
                    continue
                ax1.scatter(agent_obs["pose"][0], agent_obs["pose"][1], label=agent_id)

            ax0.set_aspect("equal")
            ax1.set_aspect("equal")

            # invert y axis
            ax0.set_ylim(ax0.get_ylim()[::-1])
            ax1.set_ylim(ax1.get_ylim()[::-1])

            plt.legend()
            plt.show()

        env.render()
