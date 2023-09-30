from typing import Dict

import numpy as np

from numba import njit

from gym_envs.multi_agent_env.common.utils import (
    get_adaptive_lookahead,
    intersect_point,
    nearest_point,
    simple_norm_axis1,
)
from gym_envs.multi_agent_env.common.track import Track, Raceline
from gym_envs.multi_agent_env.planners.planner import Planner

try:
    from pyglet.gl import GL_POINTS
except:
    pass


@njit(fastmath=False, cache=True)
def get_actuation(
    pose_theta, lookahead_point, position, lookahead_distance, wheelbase, max_steer=0.41
):
    """
    Returns actuation
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)], dtype=np.float32),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


@njit(cache=True)
def get_actuation_PD(
    pose_theta,
    lookahead_point,
    position,
    lookahead_distance,
    wheelbase,
    prev_error,
    P,
    D,
):
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)]),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    error = 2.0 * waypoint_y / lookahead_distance**2
    # radius = 1 / error
    # steering_angle = np.arctan(wheelbase / radius)
    if np.abs(waypoint_y) < 1e-4:
        return speed, 0.0, error
    steering_angle = P * error + D * (error - prev_error)
    return speed, steering_angle, error


class PurePursuitPlanner(Planner):
    def __init__(
        self,
        track: Track,
        params: dict = {},
        agent_id: str = "ego",
        render_wps_rgb: tuple = (183, 193, 222),
    ):
        super().__init__(agent_id=agent_id)

        # pure pursuit params
        self.params = {
            "lookahead_distance": 1.5,
            "max_reacquire": 20.0,
            "wb": 0.33,
            "fixed_speed": None,
            "vgain": 1.0,
            "vgain_std": 0.0,
            "min_speed": 0.0,
            "max_speed": 10.0,
            "max_steering": 1.0,
        }
        self.params.update(params)
        self.vgain = None

        self.track = track
        self.update_raceline(track.raceline)

        self.drawn_waypoints = []
        self.render_waypoints_rgb = render_wps_rgb

    def reset(self, **kwargs):
        if "vgain" in kwargs:
            assert isinstance(kwargs["vgain"], float), "vgain must be a float"
            assert kwargs["vgain"] >= 0.0, "vgain must be non-negative"
            self.params["vgain"] = kwargs["vgain"]
        if "vgain_std" in kwargs:
            assert isinstance(kwargs["vgain_std"], float), "vgain_std must be a float"
            assert kwargs["vgain_std"] >= 0.0, "vgain_std must be non-negative"
            self.params["vgain_std"] = kwargs["vgain_std"]

        mu = self.params["vgain"]
        std = self.params["vgain_std"]

        self.vgain = np.random.normal(mu, std)
        self.vgain = np.clip(self.vgain, 0.0, 1.0)  # sanity check

    def update_raceline(self, raceline: Raceline):
        self.waypoints = np.stack(
            [
                raceline.xs,
                raceline.ys,
                raceline.velxs,
            ],
            axis=1,
        )

    @staticmethod
    def _get_current_waypoint(waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((waypoints[:, 0], waypoints[:, 1])).T
        lookahead_distance = np.array(lookahead_distance, dtype=np.float32)
        _, nearest_dist, t, i = nearest_point(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = intersect_point(
                position, lookahead_distance, wpts, i + t, wrap=True
            )
            if i2 == None:
                return waypoints[-1, :]
            current_waypoint = np.empty((3,), dtype=np.float32)
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, 2]
            return current_waypoint
        else:
            return waypoints[i, :]

    def plan(
        self, observation, agent_id: str = "ego", waypoints: np.ndarray = None
    ) -> Dict[str, float]:
        if (
            agent_id is not None and agent_id in observation
        ) or "pose" not in observation:
            assert agent_id == self.agent_id, "inconsistent assignment of agent_id"
            observation = observation[self.agent_id]

        assert "pose" in observation
        pose_x, pose_y, pose_theta = observation["pose"]

        if waypoints is None:
            waypoints = self.waypoints
        waypoints = waypoints.astype(np.float32)

        position = np.array([pose_x, pose_y], dtype=np.float32)
        lookahead_point = self._get_current_waypoint(
            waypoints=waypoints,
            lookahead_distance=self.params["lookahead_distance"],
            position=position,
            theta=pose_theta,
        )

        if lookahead_point is None:
            return {"steering": 0.0, "velocity": 4.0}

        speed, steering = get_actuation(
            pose_theta=pose_theta,
            lookahead_point=lookahead_point,
            position=position,
            lookahead_distance=self.params["lookahead_distance"],
            wheelbase=self.params["wb"],
        )

        if self.params["fixed_speed"] is None:
            speed = self.vgain * speed
        else:
            speed = self.vgain

        # cap inputs to bounds
        speed = min(max(speed, self.params["min_speed"]), self.params["max_speed"])
        steering = min(
            max(steering, -self.params["max_steering"]), self.params["max_steering"]
        )

        return {"steering": steering, "velocity": speed}


class AdvancedPurePursuitPlanner(PurePursuitPlanner):
    def __init__(
        self,
        track,
        params: dict = {},
        agent_id: str = "ego",
        render_wps_rgb: tuple = (183, 193, 222),
    ):
        super().__init__(track, params, agent_id, render_wps_rgb)

        # advanced pure pursuit
        advanced_params = {
            "minL": 0.5,
            "maxL": 1.5,
            "minP": 0.4,
            "maxP": 0.6,
            "Pscale": 8.0,
            "Lscale": 8.0,
            "D": 2.0,
            "interp_scale": 10,
        }
        self.params.update(advanced_params)

        self.update_raceline(track.raceline)

        self.prev_error = 0.0
        self.drawn_waypoints = []
        self.render_waypoints_rgb = render_wps_rgb

        # ittc
        self.debug = True

    def plan(
        self, observation, agent_id: str = None, waypoints: np.ndarray = None
    ) -> Dict[str, float]:
        if agent_id is not None or "pose" not in observation:
            assert agent_id == self.agent_id, "inconsistent assignment of agent_id"
            observation = observation[self.agent_id]

        assert type(observation) == dict, "assume observation is a dict"
        assert (
            "pose" in observation
        ), "assume observation contains 'pose', instead got {}".format(observation)
        assert (
            "velocity" in observation
        ), "assume observation contains 'velocity', instead got {}".format(observation)

        pose_x, pose_y, pose_theta = observation["pose"]
        position = np.array([pose_x, pose_y])
        curr_v = observation["velocity"][0]

        if waypoints is None:
            waypoints = self.waypoints

        # get L, P with speed
        L = get_adaptive_lookahead(
            curr_v, self.params["minL"], self.params["maxL"], self.params["Lscale"]
        )
        P = (
            self.params["maxP"]
            - curr_v
            * (self.params["maxP"] - self.params["minP"])
            / self.params["Pscale"]
        )

        lookahead_point, newL, nearest_dist = self._get_interpolated_waypoint(
            waypoints, L, position, pose_theta, self.params["interp_scale"]
        )
        lookahead_point = lookahead_point.astype(np.float32)

        speed, steering, error = get_actuation_PD(
            pose_theta,
            lookahead_point,
            position,
            newL,
            self.params["wb"],
            self.prev_error,
            P,
            self.params["D"],
        )
        speed = speed * self.vgain
        self.prev_error = error

        # cap inputs to bounds
        speed = min(max(speed, 0.0), self.params["max_speed"])
        steering = min(
            max(steering, -self.params["max_steering"]), self.params["max_steering"]
        )

        return {"steering": steering, "velocity": speed}

    @staticmethod
    def _get_interpolated_waypoint(
        waypoints, lookahead_distance, position, theta, interp_scale
    ):
        n_wps = waypoints.shape[0]
        traj_distances = simple_norm_axis1(waypoints[:, :2] - position)
        nearest_idx = np.argmin(traj_distances)
        nearest_dist = traj_distances[nearest_idx]
        segment_end = nearest_idx

        if n_wps < 100 and traj_distances[n_wps - 1] < lookahead_distance:
            segment_end = n_wps - 1
        else:
            while traj_distances[segment_end] < lookahead_distance:
                segment_end = (segment_end + 1) % n_wps

        segment_begin = (segment_end - 1 + n_wps) % n_wps
        x_array = np.linspace(
            waypoints[segment_begin, 0], waypoints[segment_end, 0], interp_scale
        )
        y_array = np.linspace(
            waypoints[segment_begin, 1], waypoints[segment_end, 1], interp_scale
        )
        v_array = np.linspace(
            waypoints[segment_begin, 2], waypoints[segment_end, 2], interp_scale
        )

        xy_interp = np.vstack((x_array, y_array)).T
        dist_interp = simple_norm_axis1(xy_interp - position) - lookahead_distance
        i_interp = np.argmin(np.abs(dist_interp))
        target_global = np.array((x_array[i_interp], y_array[i_interp]))
        new_L = np.linalg.norm(position - target_global)

        lookahead_point = np.array(
            (x_array[i_interp], y_array[i_interp], v_array[i_interp])
        )
        return lookahead_point, new_L, nearest_dist


if __name__ == "__main__":
    import gymnasium as gym
    from planner import run_planner

    render = True

    # test pp controller in spielberg track
    env = gym.make("f110-multi-agent-v0", track_name="Spielberg")

    print("testing pure pursuit planner")
    pp = PurePursuitPlanner(
        env.track, params={"vgain": (0.5, 0.1)}, render_wps_rgb=(0, 255, 0)
    )
    env.add_render_callback(pp.render_waypoints)
    run_planner(env, pp, render=render)
    print()

    print("testing adaptive pure pursuit planner")
    pp = AdvancedPurePursuitPlanner(
        env.track, params={"vgain": 1.0}, render_wps_rgb=(0, 255, 0)
    )
    run_planner(env, pp, render=render)
    print()

    env.close()
