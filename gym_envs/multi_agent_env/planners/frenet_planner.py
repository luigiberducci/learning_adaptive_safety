import math
from abc import abstractmethod
from collections import namedtuple
from typing import Tuple

import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt as edt

from gym_envs.multi_agent_env.common.splines import QuinticPolynomial, QuarticPolynomial
from gym_envs.multi_agent_env.common.splines.cubic_spline import CubicSpline2D
from gym_envs.multi_agent_env.common.utils import (
    cartesian_to_frenet,
    simple_norm_axis1,
    get_vertices,
    collision,
    map_collision,
    traj_global2local,
)
from gym_envs.multi_agent_env.common.track import Track, Raceline
from gym_envs.multi_agent_env.planners.planner import Planner

# indices frenet trajectories (id starts with F to indicate frenet trajectory)
FRENET_TRAJ_DIMS = 9
FT_ID, FS_ID, FDS_ID, FDDS_ID, FDDDS_ID, FD_ID, FDD_ID, FDDD_ID, FDDDD_ID = [
    i for i in range(FRENET_TRAJ_DIMS)
]

# indices cartesian trajectories: t, x, y, vx, yaw, k, s, d
CARTESIAN_TRAJ_DIMS = 8
CT_ID, CX_ID, CY_ID, CV_ID, CYAW_ID, CK_ID, CS_ID, CD_ID = [
    i for i in range(CARTESIAN_TRAJ_DIMS)
]

# planner state
PlannerState = namedtuple(
    "PlannerState",
    [
        "t",  # time [s]
        "v",  # current velocity  [m/s]
        "s",  # current progress on track [m]
        "d_s",  # current longitudinal velocity [m/s]
        "d",  # current lateral offset from track [m]
        "d_d",  # current lateral velocity [m/s]
        "d_dd",  # current lateral acceleration [m/s^2]
        "n_v",  # next target velocity [m/s]
    ],
)


class FrenetPlanner(Planner):
    """
    Frenet Planner samples dynamically feasible trajectories in the Frenet frame
    as quintic polynomials parametrized by the terminal conditions: lat displacement d, time interval t, velocity v.

    The trajectories as sampled from a grid of d, t, v values,
    and then filtered out if violating feasibility constraints (eg., velocity, acceleration, curvature).

    The trajectories are then evaluated based on an evaluation function (e.g, min cost, max score).
    """

    def __init__(self, track: Track, params: dict = {}, agent_id: str = "ego"):
        super().__init__(agent_id=agent_id)

        self.params = {
            # sampling parameters
            "sampling_params": {
                "mode": "dtv_grid",  # "dtv_grid" means grid of lat displacement (d), time interval (t), velocity (v)
                "time_horizon": 0.5,  # [s]
                "lat_grid_lb": -1.0,  # [m]
                "lat_grid_ub": 1.0,
                "lat_grid_size": 11,
                "vel_grid_lb": 0.5,  # [m/s]
                "vel_grid_ub": 1.0,
                "vel_grid_size": 5,
                "traj_points": 30,  # nr points to discretize the trajectory
            },
            # trajectory constraints
            "max_speed": 8.0,  # not used TODO
            "vgain": 1.0,  # velocity gain to scale the raceline velocity, in range [0, 1]
            # tracking parameters
            "planning_frequency": 10,
            "tracker": "advanced_pure_pursuit",
            "tracker_params": {"vgain": 1.0},
        }
        self.params.update(params)
        self.replan = True

        # setup map info for collision check
        self.track = track
        self.dist_trans = track.spec.resolution * edt(track.occupancy_map)
        origin_x, orig_y, orig_theta = track.spec.origin
        origin_s, origin_c = np.sin(orig_theta), np.cos(orig_theta)
        self.map_metainfo = (
            origin_x,
            orig_y,
            origin_c,
            origin_s,
            *track.occupancy_map.shape,
            track.spec.resolution,
            track.track_length,
        )

        # internal variables
        self.update_raceline(track.raceline)
        self.cartesian_trajectories = None
        self.frenet_trajectories = None
        self.best_trajectory = np.zeros(
            (self.params["sampling_params"]["traj_points"], CARTESIAN_TRAJ_DIMS)
        )
        self.opp_trajectory = np.zeros(
            (self.params["sampling_params"]["traj_points"], CARTESIAN_TRAJ_DIMS)
        )
        self.planner_state = PlannerState(
            t=0.0, v=0.0, s=0.0, d_s=0.0, d=0.0, d_d=0.0, d_dd=0.0, n_v=0.0
        )
        self.opponent_state = PlannerState(
            t=0.0, v=0.0, s=0.0, d_s=0.0, d=0.0, d_d=0.0, d_dd=0.0, n_v=0.0
        )
        self.dt = 0.01  # sim time step [s]

        # tracker: used to track the best trajectory found
        self.tracker = tracker_factory(
            tracker=self.params["tracker"],
            track=self.track,
            params=self.params["tracker_params"],
            agent_id=self.agent_id,
        )
        self.time_interval = self.params["planning_frequency"] * self.dt
        self.step = 0

        # rendering
        self.draw_traj_pts = []
        self.draw_otraj_pts = []
        self.draw_grid_pts = []

    def reset(self, **kwargs):
        """
        Reset the planner and opponent state.
        :param kwargs: -
        :return: None
        """
        self.tracker.reset()
        self.time_interval = self.params["planning_frequency"] * self.dt
        self.step = 0
        self.cartesian_trajectories = None
        self.frenet_trajectories = None
        self.best_trajectory = np.zeros(
            (self.params["sampling_params"]["traj_points"], CARTESIAN_TRAJ_DIMS)
        )
        self.opp_trajectory = np.zeros(
            (self.params["sampling_params"]["traj_points"], CARTESIAN_TRAJ_DIMS)
        )
        self.planner_state = PlannerState(
            t=0.0, v=0.0, s=0.0, d_s=0.0, d=0.0, d_d=0.0, d_dd=0.0, n_v=0.0
        )
        self.opponent_state = PlannerState(
            t=0.0, v=0.0, s=0.0, d_s=0.0, d=0.0, d_d=0.0, d_dd=0.0, n_v=0.0
        )

    def update_raceline(self, raceline: Raceline):
        self.raceline_spline = raceline.spline
        self.waypoints = np.stack(
            [
                raceline.xs,
                raceline.ys,
                raceline.velxs,
                raceline.yaws,
            ],
            axis=1,
        )

    def plan(self, observation, agent_id: str = "ego"):
        assert "pose" in observation or (
            self.agent_id in observation and "pose" in observation[self.agent_id]
        ), "expected pose in obervation or multiagent observation"

        if "pose" in observation:
            observation = {"ego": observation}

        self.update_state(observation, agent_id)

        # plan
        if self.replan and self.step % self.params["planning_frequency"] == 0:
            self.frenet_planning(self.planner_state, self.opponent_state)

        self.step += 1

        # track best trajectory
        action = self.tracker.plan(
            observation, agent_id=agent_id, waypoints=self.best_traj_xyvt
        )

        return action

    def update_state(self, observation, agent_id):
        # update planner state
        pose = observation[agent_id]["pose"]
        t = observation[agent_id]["time"][0]
        v = observation[agent_id]["velocity"][0]
        s, d = cartesian_to_frenet(pose[:2], self.waypoints[:, :2])
        d_s = (s - self.planner_state.s) / self.dt
        # d_d = (d - self.planner_state.d) / self.dt if self.step > 0 else 0.0
        # d_dd = (d_d - self.planner_state.d_d) / self.dt if self.step > 0 else 0.0
        d_d = d_dd = 0.0
        next_wp, _, _ = self._get_interpolated_waypoint(
            waypoints=self.waypoints,
            lookahead_distance=0.5,
            position=pose[:2],
            theta=pose[-1],
            interp_scale=10,
        )

        n_v = next_wp[2] * self.params["vgain"]
        self.planner_state = PlannerState(
            t=t, v=v, s=s, d_s=d_s, d=d, d_d=d_d, d_dd=d_dd, n_v=n_v
        )

        # update opponent state
        opp_ids = [k for k in observation.keys() if k != agent_id]
        if len(opp_ids) > 0:
            opp_id = opp_ids[0]
            opp_pose = observation[opp_id]["pose"]
            opp_t = observation[opp_id]["time"][0]
            ov = observation[opp_id]["velocity"][0]
            os, od = cartesian_to_frenet(opp_pose[:2], self.waypoints[:, :2])
            od_s = (os - self.opponent_state.s) / self.dt
            # od_d = (od - self.opponent_state.d) / self.dt if self.step > 0 else 0.0
            # od_dd = (od_d - self.opponent_state.d_d) / self.dt if self.step > 0 else 0.0
            od_d = od_dd = 0.0
            next_owp, _, _ = self._get_interpolated_waypoint(
                waypoints=self.waypoints,
                lookahead_distance=0.5,
                position=opp_pose[:2],
                theta=pose[-1],
                interp_scale=10,
            )
            on_v = next_owp[2]
            self.opponent_state = PlannerState(
                t=opp_t, v=ov, s=os, d_s=od_s, d=od, d_d=od_d, d_dd=od_dd, n_v=on_v
            )

            assert (
                self.planner_state.t == self.opponent_state.t
            ), "expected same time for planner and opponent"

    def frenet_planning(self, planner_state, opponent_state):
        # sample many ego trajectories
        self.frenet_trajectories = self.sampling_function(planner_state)
        self.cartesian_trajectories = frenet_to_cartesian_trajectories(
            self.frenet_trajectories, self.raceline_spline
        )

        # approx opponent trajectory
        pred_time, pred_lat, pred_vgain = self.predict_opponent_intention(
            ego_state=planner_state, opponent_state=opponent_state
        )
        approx_opp_traj = sample_frenet_paths(
            opponent_state,
            lat_grid_lb=pred_lat,
            lat_grid_ub=pred_lat,
            lat_grid_size=1,
            time_horizon=pred_time,
            vel_grid_lb=pred_vgain,
            vel_grid_ub=pred_vgain,
            vel_grid_size=1,
            traj_points=self.params["sampling_params"]["traj_points"],
        )
        cartesian_opp_traj = frenet_to_cartesian_trajectories(
            approx_opp_traj, self.raceline_spline
        )
        opp_trajectories = np.repeat(
            cartesian_opp_traj, self.frenet_trajectories.shape[0], axis=0
        )

        # debug
        # self._debug_trajectories(self.cartesian_trajectories)

        all_scores, final_scores = self.evaluation_function(
            ego_trajectories=self.cartesian_trajectories,
            opp_trajectories=opp_trajectories,
            prev_ego_trajectory=self.best_trajectory,
            dt=self.dist_trans,
            map_metainfo=self.map_metainfo,
        )

        self.best_id = self.selection_function(final_scores)
        self.best_trajectory = self.cartesian_trajectories[self.best_id]
        self.opp_trajectory = cartesian_opp_traj[0]
        self.best_traj_xyvt = np.stack(
            [
                self.best_trajectory[:, CX_ID],
                self.best_trajectory[:, CY_ID],
                self.best_trajectory[:, CV_ID],
                self.best_trajectory[:, CYAW_ID],
            ],
            axis=1,
        )

    def predict_opponent_intention(
        self, ego_state: PlannerState, opponent_state: PlannerState
    ) -> np.ndarray:
        """
        Approximate opponent intention by predicting at a fixed time horizon
        its lateral position and velocity gain.

        :return: array with (time_horizon, lateral_position, velocity_gain)
        """
        pred_time = self.params["sampling_params"]["time_horizon"]
        max_lat_dev = 1.5
        pred_lat = np.clip(
            opponent_state.d + opponent_state.d_d * pred_time, -max_lat_dev, max_lat_dev
        )
        pred_vgain = opponent_state.v / self.params["max_speed"]
        return np.stack([pred_time, pred_lat, pred_vgain])

    def sampling_function(self, planner_state: np.ndarray) -> np.ndarray:
        # sampling function: used to sample trajectories in frenet frame
        params = self.params["sampling_params"]
        time_horizon = params["time_horizon"]
        lat_lb, lat_ub, lat_size = (
            params["lat_grid_lb"],
            params["lat_grid_ub"],
            params["lat_grid_size"],
        )
        vel_lb, vel_ub, vel_size = (
            params["vel_grid_lb"],
            params["vel_grid_ub"],
            params["vel_grid_size"],
        )
        traj_points = params["traj_points"]

        trajectories = sample_frenet_paths(
            planner_state,
            time_horizon,
            lat_lb,
            lat_ub,
            lat_size,
            vel_lb,
            vel_ub,
            vel_size,
            traj_points,
        )
        return trajectories

    @abstractmethod
    def evaluation_function(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        # evaluation function: used to evaluate trajectories
        raise NotImplementedError

    @abstractmethod
    def selection_function(self, scores: np.ndarray) -> int:
        # selection function: used to select the best trajectory
        raise NotImplementedError

    def render_waypoints(self, e):
        from pyglet.gl import GL_POINTS

        super().render_waypoints(e)

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = bottom - 400 + 50
        e.left = left - 400
        e.right = right + 400
        e.top = top + 400
        e.bottom = bottom - 400

        if self.cartesian_trajectories is not None:
            goal_grid_pts = np.vstack(
                [
                    self.cartesian_trajectories[:, -1, CX_ID],
                    self.cartesian_trajectories[:, -1, CY_ID],
                ]
            ).T
            scaled_grid_pts = 50.0 * goal_grid_pts
            for i in range(scaled_grid_pts.shape[0]):
                if len(self.draw_grid_pts) < scaled_grid_pts.shape[0]:
                    b = e.batch.add(
                        1,
                        GL_POINTS,
                        None,
                        (
                            "v3f/stream",
                            [scaled_grid_pts[i, 0], scaled_grid_pts[i, 1], 0.0],
                        ),
                        ("c3B/stream", [183, 193, 222]),
                    )
                    self.draw_grid_pts.append(b)
                else:
                    self.draw_grid_pts[i].vertices = [
                        scaled_grid_pts[i, 0],
                        scaled_grid_pts[i, 1],
                        0.0,
                    ]

            best_traj_pts = np.vstack(
                [self.best_trajectory[:, CX_ID], self.best_trajectory[:, CY_ID]]
            ).T
            scaled_btraj_pts = 50.0 * best_traj_pts
            for i in range(scaled_btraj_pts.shape[0]):
                if len(self.draw_traj_pts) < scaled_btraj_pts.shape[0]:
                    b = e.batch.add(
                        1,
                        GL_POINTS,
                        None,
                        (
                            "v3f/stream",
                            [scaled_btraj_pts[i, 0], scaled_btraj_pts[i, 1], 0.0],
                        ),
                        ("c3B/stream", [183, 193, 222]),
                    )
                    self.draw_traj_pts.append(b)
                else:
                    self.draw_traj_pts[i].vertices = [
                        scaled_btraj_pts[i, 0],
                        scaled_btraj_pts[i, 1],
                        0.0,
                    ]

            opp_traj_pts = np.vstack(
                [self.opp_trajectory[:, CX_ID], self.opp_trajectory[:, CY_ID]]
            ).T
            scaled_otraj_pts = 50.0 * opp_traj_pts
            for i in range(scaled_otraj_pts.shape[0]):
                if len(self.draw_otraj_pts) < scaled_otraj_pts.shape[0]:
                    b = e.batch.add(
                        1,
                        GL_POINTS,
                        None,
                        (
                            "v3f/stream",
                            [scaled_otraj_pts[i, 0], scaled_otraj_pts[i, 1], 0.0],
                        ),
                        ("c3B/stream", [222, 140, 183]),
                    )
                    self.draw_otraj_pts.append(b)
                else:
                    self.draw_otraj_pts[i].vertices = [
                        scaled_otraj_pts[i, 0],
                        scaled_otraj_pts[i, 1],
                        0.0,
                    ]

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

    def _debug_trajectories(
        self,
        ego_trajectories: np.ndarray,
        opp_trajectories: np.ndarray = None,
        norm_ego_scores: np.ndarray = None,
        zoom_in: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt
        from sim_utils.visualize_data import (
            visualize_map,
            visualize_car,
            visualize_trajectory,
        )

        plt.cla()
        ax = plt.gca()

        visualize_map(track=self.track, ax=ax)

        # plot car footprint
        ego_pose_t0 = np.array(
            [
                ego_trajectories[0, 0, CX_ID],
                ego_trajectories[0, 0, CY_ID],
                ego_trajectories[0, 0, CYAW_ID],
            ]
        )
        visualize_car(track=self.track, pose=ego_pose_t0, ax=ax)

        if opp_trajectories is not None:
            opp_pose_t0 = np.array(
                [
                    opp_trajectories[0, 0, CX_ID],
                    opp_trajectories[0, 0, CY_ID],
                    opp_trajectories[0, 0, CYAW_ID],
                ]
            )
            visualize_car(track=self.track, pose=opp_pose_t0, ax=ax)

        # plot trajectories
        opp_trajectories = (
            opp_trajectories
            if opp_trajectories is not None
            else [None] * len(ego_trajectories)
        )
        all_points = []
        for i, (ego_trajectory, opp_trajectory) in enumerate(
            zip(ego_trajectories, opp_trajectories)
        ):
            ego_xyv = ego_trajectory[:, [CX_ID, CY_ID, CV_ID]]

            if norm_ego_scores is not None:
                norm_color_array = norm_ego_scores[i].repeat(
                    ego_trajectories.shape[1] - 1
                )
            else:
                norm_color_array = None

            lc, points = visualize_trajectory(
                track=self.track,
                trajectory=ego_xyv,
                ax=ax,
                norm_color_array=norm_color_array,
            )
            all_points.extend(points)

            if opp_trajectory is not None:
                opp_xyv = opp_trajectory[:, [CX_ID, CY_ID, CV_ID]]
                lc, points = visualize_trajectory(
                    track=self.track, trajectory=opp_xyv, ax=ax
                )

        if zoom_in:
            all_points = np.array(all_points)
            minx, maxx = np.min(all_points[:, 0]), np.max(all_points[:, 0])
            miny, maxy = np.min(all_points[:, 1]), np.max(all_points[:, 1])
            ax.set_xlim(minx - 50, maxx + 50)
            ax.set_ylim(miny - 50, maxy + 50)
        plt.pause(0.001)


# @njit(cache=True)
def sample_frenet_paths(
    planner_state: np.ndarray,
    time_horizon,
    lat_grid_lb,
    lat_grid_ub,
    lat_grid_size,
    vel_grid_lb,
    vel_grid_ub,
    vel_grid_size,
    traj_points,
) -> np.ndarray:
    """
    Sample frenet paths in the Frenet frame.
    :param planner_state: current planner state (t, v, s, d_s, d, d_d, d_dd, n_v)
    :return: frenet paths
    """
    t, v, s, d_s, d, d_d, d_dd, n_v = planner_state

    # compute vel grid based on current velocity
    vel_grid = np.linspace(vel_grid_lb, vel_grid_ub, vel_grid_size) * n_v

    n_trajectories = lat_grid_size * vel_grid_size
    traj_points = traj_points
    k_frenet = FRENET_TRAJ_DIMS  # frenet params: t, s, ds, dds, ddds, d, dd, ddd, dddd

    trajectories = np.zeros((n_trajectories, traj_points, k_frenet))

    # sample lateral displacement (d)
    traj_id = 0
    for di in np.linspace(lat_grid_lb, lat_grid_ub, lat_grid_size):
        # lateral motion planning
        lat_poly = QuinticPolynomial(d, d_d, d_dd, di, 0.0, 0.0, time_horizon)

        tt = [t for t in np.linspace(0.0, time_horizon, traj_points)]
        td = [lat_poly.calc_point(t) for t in tt]
        td_d = [lat_poly.calc_first_derivative(t) for t in tt]
        td_dd = [lat_poly.calc_second_derivative(t) for t in tt]
        td_ddd = [lat_poly.calc_third_derivative(t) for t in tt]

        # sample velocity (v)
        for vi in vel_grid:
            lon_qp = QuarticPolynomial(s, v, 0.0, vi, 0.0, time_horizon)

            ts = [lon_qp.calc_point(t) for t in tt]
            td_s = [lon_qp.calc_first_derivative(t) for t in tt]
            td_ds = [lon_qp.calc_second_derivative(t) for t in tt]
            td_dds = [lon_qp.calc_third_derivative(t) for t in tt]

            trajectories[traj_id] = np.stack(
                [tt, ts, td_s, td_ds, td_dds, td, td_d, td_dd, td_ddd], axis=1
            )
            traj_id += 1

    return trajectories


def frenet_to_cartesian_trajectories(trajectories: np.ndarray, spline: CubicSpline2D):
    N, T, F = trajectories.shape

    cartesian_trajectories = np.zeros((N, T, CARTESIAN_TRAJ_DIMS))
    for i in range(N):
        trajectory = trajectories[i]

        # calc global positions
        xs, ys, vs = np.zeros(T), np.zeros(T), np.zeros(T)
        for t in range(T):
            s = trajectory[t, FS_ID]
            if not spline.s[0] < trajectory[t, FS_ID] < spline.s[-1]:
                s = (trajectory[t, FS_ID] + spline.s[-1]) % spline.s[-1]
            ix, iy = spline.calc_position(s)
            if ix is None:
                raise ValueError(
                    f"Cannot find frenet path in spline, {trajectory[t, FS_ID]}, max s: {spline.s[-1]}"
                )
            i_yaw = spline.calc_yaw(s)
            di = trajectory[t, FD_ID]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            xs[t] = fx
            ys[t] = fy
            vs[t] = trajectory[t, FDS_ID]

        # calc yaw and ds
        dxs = np.array([xs[i + 1] - xs[i] for i in range(T - 1)])
        dys = np.array([ys[i + 1] - ys[i] for i in range(T - 1)])
        yaws = np.array(
            [math.atan2(dys[i], dxs[i]) for i in range(T - 1)]
            + [math.atan2(dys[-1], dxs[-1])]
        )
        ds = np.array(
            [math.hypot(dxs[i], dys[i]) for i in range(T - 1)]
            + [math.hypot(dxs[-1], dys[-1])]
        )

        # calc curvature
        ks = np.array(
            [(yaws[i + 1] - yaws[i]) / (ds[i] + 1e-6) for i in range(T - 1)] + [0.0]
        )

        cartesian_trajectories[i] = np.stack(
            [
                trajectory[:, FT_ID],
                xs,
                ys,
                vs,
                yaws,
                ks,
                trajectory[:, FS_ID],
                trajectory[:, FD_ID],
            ],
            axis=1,
        )

    return cartesian_trajectories


def tracker_factory(
    tracker: str, track: Track, params: dict, agent_id: str
) -> callable:
    if tracker == "pure_pursuit":
        from gym_envs.multi_agent_env.planners.pure_pursuit import PurePursuitPlanner

        return PurePursuitPlanner(track, params, agent_id)
    elif tracker == "advanced_pure_pursuit":
        from gym_envs.multi_agent_env.planners.pure_pursuit import (
            AdvancedPurePursuitPlanner,
        )

        return AdvancedPurePursuitPlanner(track, params, agent_id)
    else:
        raise NotImplementedError(f"Tracker not implemented for {tracker}")


class FrenetPlannerWeightedCosts(FrenetPlanner):
    def __init__(self, track: Track, params: dict = {}, agent_id: str = "ego"):
        super().__init__(track=track, params=params, agent_id=agent_id)

        self.params.update(
            {
                "cost_weights": {
                    "get_curvature_cost": 1.0,
                    "get_length_cost": 1.0,
                    "get_similarity_cost": 0.5,
                    "get_follow_optim_cost": 1.0,
                    "get_map_collision": 1.0,
                    "get_v_cost": 3.0,
                    "get_opponent_collision": 1.0,
                },
            }
        )
        self.params.update(params)

        self.cost_fns = [
            cost_fn_factory(fn_name)
            for fn_name, w in self.params["cost_weights"].items()
        ]
        self.weights = [w for fn_name, w in self.params["cost_weights"].items()]

    def reset(self, **kwargs):
        super().reset(**kwargs)
        if "cost_weights" in kwargs:
            if isinstance(kwargs["cost_weights"], list) or isinstance(
                kwargs["cost_weights"], np.ndarray
            ):
                old_weight_dict = self.params["cost_weights"]
                self.params["cost_weights"] = {
                    n: kwargs["cost_weights"][i]
                    for i, (n, _) in enumerate(old_weight_dict.items())
                }
                self.weights = kwargs["cost_weights"]
            else:
                self.params["cost_weights"] = kwargs["cost_weights"]
        if "vgain" in kwargs:
            self.params["vgain"] = kwargs["vgain"]

    def evaluation_function(
        self,
        ego_trajectories: np.ndarray,
        opp_trajectories: np.ndarray,
        prev_ego_trajectory: np.ndarray,
        dt: np.ndarray,
        map_metainfo: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert len(self.weights) == len(
            self.cost_fns
        ), "Weights and cost functions must have the same length"

        all_scores = np.zeros((ego_trajectories.shape[0], len(self.cost_fns)))

        for i, cost_fn in enumerate(self.cost_fns):
            all_scores[:, i] = cost_fn(
                ego_trajectories,
                opp_trajectories,
                prev_ego_trajectory,
                dt,
                map_metainfo,
            )

        final_scores = np.sum(all_scores * self.weights, axis=1)

        return all_scores, final_scores

    def selection_function(self, scores: np.ndarray) -> int:
        best_ids = np.where(scores == np.min(scores))[0]
        return np.random.choice(best_ids)


class ControllableFrenetPlanner(FrenetPlanner):
    """
    This frenet planner can be controlled externally by setting the
    goal as (d_lat, vgain) and it will follow a trajectory produced by
    fitting a frenet trajectory to the goal and then tracking it with
    a pure pursuit controller.
    """

    def __init__(self, track: Track, params: dict = {}, agent_id: str = "ego"):
        super().__init__(track=track, params=params, agent_id=agent_id)

        self.action_dgoal = 0.0
        self.action_vgain = 1.0

    def set_action(self, dgoal: float, vgain: float):
        self.action_dgoal = np.float32(dgoal)
        self.action_vgain = np.float32(vgain)

    def sampling_function(self, planner_state: np.ndarray) -> np.ndarray:
        # sampling function: used to sample trajectories in frenet frame
        params = self.params["sampling_params"]
        time_horizon = params["time_horizon"]
        lat_lb, lat_ub, lat_size = self.action_dgoal, self.action_dgoal, 1
        vel_lb, vel_ub, vel_size = self.action_vgain, self.action_vgain, 1
        traj_points = params["traj_points"]

        trajectories = sample_frenet_paths(
            planner_state,
            time_horizon,
            lat_lb,
            lat_ub,
            lat_size,
            vel_lb,
            vel_ub,
            vel_size,
            traj_points,
        )
        return trajectories

    def evaluation_function(
        self,
        ego_trajectories: np.ndarray,
        opp_trajectories: np.ndarray,
        prev_ego_trajectory: np.ndarray,
        dt: np.ndarray,
        map_metainfo: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """dummy evaluation because single trajectory is always selected"""
        return np.zeros(ego_trajectories.shape[0]), np.zeros(ego_trajectories.shape[0])

    def selection_function(self, scores: np.ndarray) -> int:
        return 0


"""
cost functions
"""


def cost_fn_factory(fn_name: str) -> callable:
    if fn_name == "get_curvature_cost":
        return get_curvature_cost
    elif fn_name == "get_length_cost":
        return get_length_cost
    elif fn_name == "get_similarity_cost":
        return get_similarity_cost
    elif fn_name == "get_follow_optim_cost":
        return get_follow_optim_cost
    elif fn_name == "get_map_collision":
        return get_map_collision
    elif fn_name == "get_v_cost":
        return get_v_cost
    elif fn_name == "get_opponent_collision":
        return get_opponent_collision
    else:
        raise ValueError("Unknown cost function name: {}".format(fn_name))


@njit(cache=True)
def get_length_cost(
    ego_trajectories: np.ndarray,
    opp_trajectories: np.ndarray,
    prev_trajectory: np.ndarray,
    dt: np.ndarray,
    map_metainfo: np.array,
) -> np.ndarray:
    scale = 0.5

    costs = np.zeros(ego_trajectories.shape[0])

    for i in range(ego_trajectories.shape[0] - 1):
        dx = ego_trajectories[i + 1, :, CX_ID] - ego_trajectories[i, :, CX_ID]
        dy = ego_trajectories[i + 1, :, CY_ID] - ego_trajectories[i, :, CY_ID]
        costs[i] = np.sum(np.sqrt(dx * dx + dy * dy))

    return scale * costs


# @njit(cache=True)
def get_similarity_cost(
    ego_trajectories: np.ndarray,
    opp_trajectories: np.ndarray,
    prev_trajectory: np.ndarray,
    dt: np.ndarray,
    map_metainfo: np.array,
) -> np.ndarray:
    scale = 0.15

    costs = np.zeros(ego_trajectories.shape[0])

    # first iteration: prev trajectory initialized to zero
    if np.sum(prev_trajectory) < 1e-6:
        return costs

    # assume all trajectories start at the same pose
    ego_pose = np.array(
        [
            ego_trajectories[0, 0, CX_ID],
            ego_trajectories[0, 0, CY_ID],
            ego_trajectories[0, 0, CYAW_ID],
        ]
    )

    prev_traj_xy = np.stack(
        [prev_trajectory[:, CX_ID], prev_trajectory[:, CY_ID]], axis=1
    )
    ego_traj_xy = np.stack(
        [ego_trajectories[:, :, CX_ID], ego_trajectories[:, :, CY_ID]], axis=2
    )

    prev_local_traj = traj_global2local(ego_pose, prev_traj_xy)
    local_traj = traj_global2local(ego_pose, ego_traj_xy)
    diff = local_traj - prev_local_traj
    cost = diff * diff
    cost = np.sum(cost, axis=(1, 2)) * scale
    return cost


@njit(cache=True)
def get_map_collision(
    ego_trajectories: np.ndarray,
    opp_trajectories: np.ndarray,
    prev_trajectory: np.ndarray,
    dt: np.ndarray,
    map_metainfo: np.array,
) -> np.ndarray:
    collision_thres = 0.3
    collision_cost = 3000

    if dt is None:
        raise ValueError(
            "Map Distance Transform dt has to be set when using this cost function."
        )
    # points: (n, 2)
    ego_trajectories = ego_trajectories[
        :, :, CX_ID : CY_ID + 1
    ]

    all_traj_pts = np.ascontiguousarray(ego_trajectories).reshape(-1, 2)  # (nxm, 2)
    collisions = map_collision(
        all_traj_pts, dt, map_metainfo, eps=collision_thres
    )  # (nxm)
    collisions = collisions.reshape(ego_trajectories.shape[0], -1)  # (n, m)
    cost = []
    for traj_collision in collisions:
        c = 0
        if np.any(traj_collision):
            c = collision_cost
        cost.append(c)

    return np.array(cost)


@njit(cache=True)
def get_follow_optim_cost(
    ego_trajectories: np.ndarray,
    opp_trajectories: np.ndarray,
    prev_trajectory: np.ndarray,
    dt: np.ndarray,
    map_metainfo: np.array,
) -> np.ndarray:
    """
    It returns the lateral displacement between the trajectory endpoint and the raceline.
    Since the cost will be within 0 and half track-width (approx. 1.5m), it is not scaled.
    """
    scale = 1.0

    cost = np.array(
        [
            np.abs(ego_trajectories[i, -1, CD_ID])
            for i in range(ego_trajectories.shape[0])
        ]
    )

    return cost * scale


# @njit(cache=True)
def get_curvature_cost(
    ego_trajectories: np.ndarray,
    opp_trajectories: np.ndarray,
    prev_trajectory: np.ndarray,
    dt: np.ndarray,
    map_metainfo: np.array,
) -> np.ndarray:
    scale = 1.0

    costs = np.zeros(ego_trajectories.shape[0])

    for i in range(ego_trajectories.shape[0]):
        costs[i] = np.mean(np.abs(ego_trajectories[i, :, CK_ID]))

    return scale * costs


@njit(cache=True)
def get_v_cost(
    ego_trajectories: np.ndarray,
    opp_trajectories: np.ndarray,
    prev_trajectory: np.ndarray,
    dt: np.ndarray,
    map_metainfo: np.array,
) -> np.ndarray:
    scale_v = 2.0
    scale_kv = 1.0

    costs = np.zeros(ego_trajectories.shape[0])

    mean_v = np.array(
        [
            np.mean(ego_trajectories[i, :, CV_ID])
            for i in range(ego_trajectories.shape[0])
        ]
    )
    mean_k = np.array(
        [
            np.mean(np.abs(ego_trajectories[i, :, CK_ID]))
            for i in range(ego_trajectories.shape[0])
        ]
    )
    all_traj_min_mean_k = np.min(mean_k)
    all_traj_max_v = np.max(ego_trajectories[:, :, CV_ID])

    for i in range(ego_trajectories.shape[0]):
        costs[i] = (
            scale_v * (all_traj_max_v - mean_v[i])
            + scale_kv * (mean_k[i] - all_traj_min_mean_k) * mean_v[i]
        )

    return costs


@njit(cache=True)
def get_opponent_collision(
    ego_trajectories: np.ndarray,
    opp_trajectories: np.ndarray,
    prev_trajectory: np.ndarray,
    dt: np.ndarray,
    map_metainfo: np.array,
) -> np.ndarray:
    v_length, v_width = 0.58, 0.31
    inflation_perc = 1.0
    collision_cost = 3000

    v_length = v_length * (1 + inflation_perc)
    v_width = v_width * (1 + inflation_perc)

    scores = np.zeros(ego_trajectories.shape[0])

    # use np.where to deal with zero padding at the end of trajectories
    traj_norm = np.zeros((ego_trajectories.shape[0], ego_trajectories.shape[1]))
    for i in range(ego_trajectories.shape[0]):
        for t in range(ego_trajectories.shape[1]):
            xy_diff = np.array(
                [
                    ego_trajectories[i, t, CX_ID] - opp_trajectories[i, t, CX_ID],
                    ego_trajectories[i, t, CY_ID] - opp_trajectories[i, t, CY_ID],
                ]
            )
            if np.linalg.norm(xy_diff) < 1e-6:
                traj_norm[i][t] = 1e6
            else:
                traj_norm[i][t] = np.linalg.norm(xy_diff)

    for i in range(ego_trajectories.shape[0]):
        close_p_idx = np.argmin(traj_norm[i])  # find the closest point

        ego_xyt = np.array(
            [
                ego_trajectories[i, close_p_idx, CX_ID],
                ego_trajectories[i, close_p_idx, CY_ID],
                ego_trajectories[i, close_p_idx, CYAW_ID],
            ]
        )
        opp_xyt = np.array(
            [
                opp_trajectories[i, close_p_idx, CX_ID],
                opp_trajectories[i, close_p_idx, CY_ID],
                opp_trajectories[i, close_p_idx, CYAW_ID],
            ]
        )

        ego_box = get_vertices(ego_xyt, v_length, v_width)

        opp_box = get_vertices(opp_xyt, v_length, v_width)

        if collision(ego_box, opp_box):
            scores[i] = collision_cost

    return scores


if __name__ == "__main__":
    import gymnasium as gym
    import gym_envs
    from gym_envs.multi_agent_env.planners.planner import Planner, run_planner
    from gym_envs.multi_agent_env.planners.pure_pursuit import PurePursuitPlanner

    np.random.seed(0)

    track_name = "Spielberg"
    track = Track.from_track_name(track_name)
    pp = PurePursuitPlanner(track=track, agent_id="npc0")

    planner = FrenetPlannerWeightedCosts(track, params={"vgain": 0.8}, agent_id="ego")
    env = gym.make("f110-multi-agent-v0", track_name=track_name, npc_planners=[pp])

    env.add_render_callback(planner.render_waypoints)

    for i in range(10):
        run_planner(
            env, planner, render=True, max_steps=1000, reset_mode="hardcurve_back"
        )
    env.close()
