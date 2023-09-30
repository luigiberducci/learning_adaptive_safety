from pyclothoids import Clothoid
import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt as edt

from gym_envs.multi_agent_env.planners.planner import Planner
from gym_envs.multi_agent_env.planners.pure_pursuit import (
    AdvancedPurePursuitPlanner,
    PurePursuitPlanner,
)
from gym_envs.multi_agent_env.common.utils import (
    nearest_point,
    get_adaptive_lookahead,
    intersect_point,
    get_rotation_matrix,
    x2y_distances_argmin,
    get_vertices,
    collision,
    map_collision,
    zero_2_2pi,
)
from gym_envs.multi_agent_env.common.track import Track, Raceline


class LatticePlanner(Planner):
    def __init__(
        self,
        track: Track,
        params: dict = {},
        agent_id: str = "ego",
        render_wps_rgb: tuple = (183, 193, 222),
    ):
        super().__init__(agent_id=agent_id)
        self.params = {
            "wb": 0.33,
            "lh_grid_lb": 0.6,
            "lh_grid_ub": 1.2,
            "lh_grid_rows": 3,
            "lat_grid_lb": -1.5,
            "lat_grid_ub": 1.5,
            "lat_grid_cols": 11,
            "weights": 7,
            "score_names": [
                "curvature_cost",
                "get_length_cost",
                "get_similarity_cost",
                "get_follow_optim_cost",
                "get_map_collision",
                "abs_v_cost",
                "collision_cost",
            ],
            # "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "traj_v_span_min": 0.2,
            "traj_v_span_max": 1.0,
            "traj_v_span_num": 5,
            "traj_points": 20,
            "vgain": 0.9,
            "collision_thres": 0.3,
            "planning_frequency": 10,
            "tracker": "advanced_pure_pursuit",
            "tracker_params": {"vgain": 1.0},
        }
        self.params.update(params)

        # load waypoints
        self.track = track  # for debugging trajectories
        self.update_raceline(track.raceline)

        # sample and cost function
        self.sample_func = None
        self.add_sample_function(sample_lookahead_square)

        self.selection_func = None

        self.shape_cost_funcs = []
        self.constant_cost_funcs = []

        self.add_shape_cost_function(get_length_cost)
        self.add_shape_cost_function(get_similarity_cost)
        self.add_shape_cost_function(get_follow_optim_cost)

        self.add_constant_cost_function(get_map_collision)

        self.set_cost_weights(self.params["weights"])

        self.v_lattice_span = np.linspace(
            self.params["traj_v_span_min"],
            self.params["traj_v_span_max"],
            self.params["traj_v_span_num"],
            dtype=np.float32,
        )
        self.v_lattice_num = self.params["traj_v_span_num"]
        self.vgain = self.params["vgain"]

        self.best_traj = None
        self.best_traj_ref_v = 0.0
        self.best_traj_idx = 0
        self.all_traj = None
        self.all_traj_clothoids = None
        self.lattice_metainfo = None

        self.prev_traj_local = np.zeros((self.params["traj_points"], 2))
        self.prev_opp_pose = np.array([0, 0])

        self.goal_grid = None
        self.step_all_cost = {}
        self.all_costs = None
        self.time_interval = self.params["planning_frequency"] * 0.01
        self.step = 0

        # tracker
        tracker_params = self.params["tracker_params"]
        self.tracker: PurePursuitPlanner = None
        if self.params["tracker"] == "pure_pursuit":
            self.tracker = PurePursuitPlanner(
                track=track, agent_id=agent_id, params=tracker_params
            )
        elif self.params["tracker"] == "advanced_pure_pursuit":
            self.tracker = AdvancedPurePursuitPlanner(
                track=track, agent_id=agent_id, params=tracker_params
            )
        else:
            raise ValueError(f'Tracker {self.params["tracker"]} not supported')

        # setup map info for collision check
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

        # create maps
        self.dt = track.spec.resolution * edt(track.occupancy_map)  # distance transform

        # render
        self.draw_traj_pts = []
        self.draw_grid_pts = []
        self.render_raceline_rgb = render_wps_rgb

    def reset(self, **kwargs):
        self.best_traj = None
        self.best_traj_ref_v = 0.0
        self.best_traj_idx = 0
        self.all_traj = None
        self.all_traj_clothoids = None
        self.lattice_metainfo = None

        self.prev_traj_local = np.zeros((self.params["traj_points"], 2))
        self.prev_opp_pose = np.array([0, 0])

        self.goal_grid = None
        self.step_all_cost = {score: None for score in self.params["score_names"]}
        self.all_costs = None
        self.time_interval = self.params["planning_frequency"] * 0.01

        # tracker
        self.tracker.reset()

    def update_raceline(self, raceline: Raceline):
        self.waypoints = np.stack(
            [
                raceline.xs,
                raceline.ys,
                raceline.velxs,
                raceline.yaws,
            ],
            axis=1,
        ).astype(np.float32)

    def add_shape_cost_function(self, func):
        if type(func) is list:
            self.shape_cost_funcs.extend(func)
        else:
            self.shape_cost_funcs.append(func)

    def add_constant_cost_function(self, func):
        self.constant_cost_funcs.append(func)

    def set_cost_weights(self, cost_weights):
        if type(cost_weights) == int:
            n = cost_weights
            cost_weights = np.array([1 / n] * n)
        self.cost_weights = cost_weights

    def add_sample_function(self, func):
        """
        Add a custom sample function to create goal grid
        """
        self.sample_func = func

    def add_selection_function(self, func):
        """
        Add a custom selection fucntion to select a trajectory. The selection function returns the index of the 'best' cost.
        """
        self.selection_func = func

    def sample(
        self, pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid, lat_grid
    ):
        """
        Sample a goal grid based on sample function. Given the current vehicle state, return a list of [x, y, theta] tuples

        Args:
            None

        Returns:
            goal_grid (numpy.ndarray [N, 3]): list of goal states, columns are [x, y, theta]
        """
        if self.sample_func is None:
            raise NotImplementedError("Please set a sample function before sampling.")

        goal_grid = self.sample_func(
            pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid, lat_grid
        )

        return goal_grid

    def eval(self, all_traj, all_traj_clothoid, opp_poses, ego_pose):
        cost_weights = self.cost_weights
        n, k = all_traj.shape[0], self.v_lattice_num

        # assume use the **first** weight to calculate curvature cost
        mean_k, max_k = get_curvature(all_traj, all_traj_clothoid)  # (n, )
        cost = 2.0 * cost_weights[0] * mean_k
        self.step_all_cost["curvature_cost"] = cost.copy()

        ## other shape cost, current include length, similarity, map collision,
        for i, func in enumerate(self.shape_cost_funcs):
            self.lattice_metainfo = (
                self.params["lh_grid_rows"],
                self.params["lat_grid_cols"],
                self.params["traj_v_span_num"],
            )
            cur_cost = func(
                all_traj,
                all_traj_clothoid,
                opp_poses,
                ego_pose,
                self.prev_traj_local,
                self.dt,
                self.map_metainfo,
                self.lattice_metainfo,
            )
            cur_cost = cost_weights[i + 1] * cur_cost
            self.step_all_cost[func.__name__] = cur_cost
            cost += cur_cost

        ## cost functions with constant scale
        for i, func in enumerate(self.constant_cost_funcs):
            cur_cost = func(
                all_traj,
                all_traj_clothoid,
                opp_poses,
                ego_pose,
                self.prev_traj_local,
                self.dt,
                self.map_metainfo,
                self.params["collision_thres"],
            )
            self.step_all_cost[func.__name__] = cur_cost
            cost += cur_cost

        ## velocity cost, assume use the last two weight to calculate abs velocity cost
        all_traj_min_mean_k = np.min(mean_k)
        mean_k_lattice = np.repeat(mean_k, k).reshape(n, k)  # (n, k)
        all_traj_v = all_traj[:, -1, 2]  # (n, )
        traj_v_lattice = (
            np.repeat(all_traj_v, k).reshape(n, k) * self.v_lattice_span * self.vgain
        )

        abs_v_cost = (
            cost_weights[-3] * (np.max(traj_v_lattice) + 1 - traj_v_lattice)
            + cost_weights[-2] * (mean_k_lattice - all_traj_min_mean_k) * traj_v_lattice
        )

        self.step_all_cost["abs_v_cost"] = abs_v_cost

        ## collision cost
        collision_cost = cost_weights[-1] * get_obstacle_collision_with_v(
            all_traj,
            all_traj_clothoid,
            traj_v_lattice,
            opp_poses,
            self.prev_opp_pose,
            self.time_interval,
        )
        self.step_all_cost["collision_cost"] = collision_cost

        cost = np.repeat(cost, k).reshape(n, k)
        cost = cost + abs_v_cost + collision_cost

        # for debugging
        all_costs = cost
        final_scores = all_costs[:, -1] if len(all_costs.shape) > 1 else all_costs
        score_arrays = np.asarray(
            [
                ss if len(ss.shape) == 1 else ss[:, -1]
                for ss in self.step_all_cost.values()
            ]
        ).T
        # self._debug_trajectories_scores(observation=observation, trajs=all_traj, final_scores=final_scores, score_arrays=score_arrays)

        return cost

    def select(self, all_costs):
        """
        Select the best trajectory based on the selection function, defaults to argmin if no custom function is defined.
        """
        if self.selection_func is None:
            self.selection_func = np.argmin
        best_idx = self.selection_func(all_costs)
        return best_idx

    def plan(self, observation, agent_id: str = "ego"):
        assert "pose" in observation or (
            self.agent_id in observation and "pose" in observation[self.agent_id]
        ), "expected pose in observation or multiagent observation"

        if "pose" in observation:
            observation = {agent_id: observation}

        step = int(observation[self.agent_id]["time"] // 0.01)
        if step % self.params["planning_frequency"] == 0:
            self.clothoid_planning(observation, agent_id)

        # track best trajectory
        action = self.tracker.plan(
            observation, agent_id=agent_id, waypoints=self.best_traj
        )

        return action

    def clothoid_planning(self, observation, agent_id: str):
        assert agent_id == self.agent_id, "inconsistent agent id in clothoid planner"

        # process state
        pose_x, pose_y, pose_theta = observation[self.agent_id]["pose"].astype(
            np.float32
        )
        velocity = observation[self.agent_id]["velocity"][0]
        opp_poses = [
            np.concatenate(
                [
                    observation[opponent_id]["pose"],
                    observation[opponent_id]["frenet_coords"],
                ]
            )
            for opponent_id in observation
            if opponent_id != self.agent_id
        ]
        opp_poses = np.array(opp_poses).reshape(-1, 5)
        waypoints = self.waypoints

        # state
        ego_pose = np.array([pose_x, pose_y, pose_theta])
        _, _, t, nearest_i = nearest_point(ego_pose[:2], waypoints[:, 0:2])

        # sample a grid based on current states
        minL, maxL, Lscale = 0.5, 1.5, 8.0
        min_L = get_adaptive_lookahead(velocity, minL, maxL, Lscale)
        lh_grid = np.linspace(
            min_L + self.params["lh_grid_lb"],
            min_L + self.params["lh_grid_ub"],
            self.params["lh_grid_rows"],
            dtype=np.float32,
        )
        lat_grid = np.linspace(
            self.params["lat_grid_lb"],
            self.params["lat_grid_ub"],
            self.params["lat_grid_cols"],
            dtype=np.float32,
        )
        self.goal_grid = self.sample(
            pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid, lat_grid
        )

        # generate clothoids
        all_traj = []
        all_traj_clothoid = []
        for point in self.goal_grid:
            clothoid = Clothoid.G1Hermite(
                pose_x, pose_y, pose_theta, point[0], point[1], point[2]
            )
            traj = sample_traj(clothoid, self.params["traj_points"], point[3])
            all_traj.append(traj)
            # G1Hermite parameters are [xstart, ystart, thetastart, curvrate, kappastart, arclength]
            all_traj_clothoid.append(np.array(clothoid.Parameters))

        # evaluate all trajectory on all costs
        all_traj = np.array(all_traj)
        all_traj_clothoid = np.array(all_traj_clothoid)
        all_costs = self.eval(all_traj, all_traj_clothoid, opp_poses, ego_pose=ego_pose)
        self.all_costs = all_costs

        # select best trajectory
        best_traj_idx = self.select(all_costs)
        row_idx, col_idx = divmod(best_traj_idx, self.v_lattice_num)
        self.best_traj_idx = best_traj_idx
        self.best_traj = all_traj[row_idx].copy()
        self.all_traj = all_traj
        self.all_traj_clothoids = all_traj_clothoid
        self.best_traj_ref_v = self.best_traj[-1, 2]
        self.best_traj[:, 2] *= self.v_lattice_span[col_idx] * self.vgain
        self.prev_traj_local = traj_global2local(ego_pose, self.best_traj[:, :2])

        if opp_poses.shape[0] > 0:
            self.prev_opp_pose = opp_poses[:, :2]

    def render_waypoints(self, e):
        """
        Custom render call back function for Lattice Planner Example

        Args:
            e: environment renderer
        """
        from pyglet.gl import GL_POINTS

        super().render_waypoints(e)

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 400
        e.right = right + 400
        e.top = top + 400
        e.bottom = bottom - 400

        if self.goal_grid is not None:
            goal_grid_pts = np.vstack([self.goal_grid[:, 0], self.goal_grid[:, 1]]).T
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

            best_traj_pts = np.vstack([self.best_traj[:, 0], self.best_traj[:, 1]]).T
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

    def _debug_trajectories_scores(
        self,
        trajs: np.ndarray,
        final_scores: np.ndarray,
        score_arrays: np.ndarray,
        ego_pose: np.ndarray,
        opp_poses: np.ndarray,
    ):
        from matplotlib import pyplot as plt
        from sim_utils.generate_clothoids import show_clothoids

        assert trajs.shape[0] == final_scores.shape[0]
        assert trajs.shape[0] == score_arrays.shape[0]

        # visualization
        track = self.track

        if len(opp_poses) > 0:
            opp_pose = opp_poses[0]
            poses = np.stack([ego_pose, opp_pose], axis=0)
        else:
            poses = ego_pose[None]

        n_scores = score_arrays.shape[1]
        fig, axs = plt.subplots(1, n_scores + 1, figsize=(16, 4))

        for i_score in range(n_scores + 1):
            if i_score == n_scores:
                score_batch = final_scores
                title = "Final score"
            else:
                score_batch = score_arrays[:, i_score]
                title = f"Score {i_score} - {self.params['score_names'][i_score]}"

            min_score, max_score = np.min(score_batch), np.max(score_batch)
            norm_scores = (score_batch - min_score) / (max_score - min_score + 1e-6)
            norm_scores = np.clip(norm_scores, 0.0, 1.0)

            color_array = norm_scores[None].T.repeat(trajs.shape[1] - 1, 1)

            ax = axs[i_score]
            show_clothoids(poses, trajs, track=track, ax=ax, norm_color=color_array)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

        plt.show()


class ControllableLatticePlanner(LatticePlanner):
    """
    This lattice planner can be controlled externally by setting the
    goal as (d_lat, vgain) and it will follow a trajectory produced by
    fitting a clothoid trajectory to the goal and then tracking it with
    a pure pursuit controller.
    """

    def __init__(self, track: Track, params: dict = {}, agent_id: str = "ego"):
        super().__init__(track=track, params=params, agent_id=agent_id)

        self.action_dgoal = 0.0
        self.action_vgain = 1.0

        # remove velocity span, direct control
        self.v_lattice_span = np.array([1.0])

    def set_action(self, dgoal: float, vgain: float):
        self.action_dgoal = dgoal
        self.action_vgain = vgain

    def sample(
        self, pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid, lat_grid
    ):
        # override original sampling function, returning a single goal point
        minL, maxL, Lscale = 0.5, 1.5, 8.0
        lh_grid = np.array([get_adaptive_lookahead(velocity, minL, maxL, Lscale)])
        lat_grid = np.array([self.action_dgoal], dtype=np.float32)

        # print("vel: ", velocity, "lookahead: ", lh_grid)

        # override vgain
        self.vgain = np.float32(self.action_vgain * self.params["vgain"])

        goal_grid = self.sample_func(
            pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid, lat_grid
        )
        return goal_grid

    def eval(self, all_traj, all_traj_clothoid, opp_poses, ego_pose):
        return np.zeros(all_traj.shape[0])


"""

Example function for sampling a grid of goal points

"""


def sample_traj(clothoid, npts, v):
    # traj (m, 5)
    traj = np.empty((npts, 5))
    k0 = clothoid.Parameters[3]
    dk = clothoid.Parameters[4]

    for i in range(npts):
        s = i * (clothoid.length / max(npts - 1, 1))
        traj[i, 0] = clothoid.X(s)
        traj[i, 1] = clothoid.Y(s)
        traj[i, 2] = v
        traj[i, 3] = clothoid.Theta(s)
        traj[i, 4] = np.sqrt(clothoid.XDD(s) ** 2 + clothoid.YDD(s) ** 2)

    return traj


@njit(cache=True)
def sample_lookahead_square(
    pose_x, pose_y, pose_theta, velocity, waypoints, lookahead_distances, widths
):
    """
    Sample a grid of goal points in front of the vehicle along the raceline.
    """
    # get lookahead points to create grid along waypoints
    position = np.array([pose_x, pose_y])
    nearest_p, nearest_dist, t, nearest_i = nearest_point(position, waypoints[:, 0:2])
    local_span = np.vstack((np.zeros_like(widths), widths))
    xy_grid = np.zeros((2, 1))
    theta_grid = np.zeros((len(lookahead_distances), 1))
    v_grid = np.zeros((len(lookahead_distances), 1))
    for i, d in enumerate(lookahead_distances):
        lh_pt, i2, t2 = intersect_point(
            np.ascontiguousarray(nearest_p),
            d,
            waypoints[:, 0:2],
            t + nearest_i,
            wrap=True,
        )
        if i2 is None:
            i2 = -1
        else:
            i2 = int(i2)
        # different heading
        lh_pt_theta = np.float32(waypoints[i2, 3] + 0.5 * np.pi)
        lh_pt_v = waypoints[i2, 2]
        lh_span_points = get_rotation_matrix(lh_pt_theta) @ local_span + lh_pt.reshape(
            2, -1
        )
        xy_grid = np.hstack((xy_grid, lh_span_points))
        theta_grid[i] = zero_2_2pi(lh_pt_theta)
        v_grid[i] = lh_pt_v
    xy_grid = xy_grid[:, 1:]
    theta_grid = np.repeat(theta_grid, len(widths)).reshape(1, -1)
    v_grid = np.repeat(v_grid, len(widths)).reshape(1, -1)
    grid = np.vstack((xy_grid, theta_grid, v_grid)).T
    return grid


"""

Tool function(maybe move to util.py afterwards)

"""


@njit(cache=True)
def traj_global2local(ego_pose, traj):
    """
    traj: (n, m, 2) or (m, 2)
    """
    new_traj = np.zeros_like(traj)
    pose_x, pose_y, pose_theta = ego_pose
    c = np.cos(pose_theta)
    s = np.sin(pose_theta)
    new_traj[..., 0] = c * (traj[..., 0] - pose_x) + s * (
        traj[..., 1] - pose_y
    )  # (n, m, 1)
    new_traj[..., 1] = -s * (traj[..., 0] - pose_x) + c * (
        traj[..., 1] - pose_y
    )  # (n, m, 1)
    return new_traj


"""

Example functions for different costs

traj: np.ndarray, (n, m, 5), (x, y, v, heading, arc_length)
traj_clothoid: np.ndarray, (n, 6), (x0, y0, theta0, k0, dk, arc_length)
opp_poses: np.ndarray, (k, 3)

Return: 
cost: np.ndarray, (n, 1) 
"""


@njit(cache=True)
def get_follow_optim_cost(
    traj,
    traj_clothoid,
    opp_poses=None,
    ego_pose=None,
    prev_traj=None,
    dt=None,
    map_metainfo=None,
    lattice_metainfo=None,
):
    scale = 0.1

    rows, cols, _ = lattice_metainfo
    idx_diff = np.empty(0)

    for _ in range(rows):
        single_row = np.arange(
            -(cols // 2), cols // 2 + 1
        )  # [-k, ..., 1, 0, 1, ..., k]
        idx_diff = np.hstack(
            (idx_diff, single_row)
        )  # concatenate to reach the shape (rows * cols,)

    cost = idx_diff * idx_diff

    return cost * scale


def get_curvature(traj, traj_clothoid):
    k0 = traj_clothoid[:, 3].reshape(-1, 1)  # (n, 1)
    dk = traj_clothoid[:, 4].reshape(-1, 1)  # (n, 1)
    s = traj_clothoid[:, -1]  # (n, )
    s_pts = np.linspace(np.zeros_like(s), s, num=traj.shape[1]).T  # (n, m)
    traj_k = k0 + dk * s_pts  # (n, m)
    traj_k = np.abs(traj_k)
    max_k = np.max(traj_k, axis=1)
    mean_k = np.mean(traj_k, axis=1)
    return mean_k, max_k


def get_all_curvature(traj, traj_clothoid):
    k0 = traj_clothoid[:, 3].reshape(-1, 1)  # (n, 1)
    dk = traj_clothoid[:, 4].reshape(-1, 1)  # (n, 1)
    s = traj_clothoid[:, -1]  # (n, )
    s_pts = np.linspace(np.zeros_like(s), s, num=traj.shape[1]).T  # (n, m)
    traj_k = k0 + dk * s_pts  # (n, m)
    return traj_k.reshape(traj_k.shape[0], traj_k.shape[1], 1)


@njit(cache=True)
def get_length_cost(
    traj,
    traj_clothoid,
    opp_poses=None,
    ego_pose=None,
    prev_traj=None,
    dt=None,
    map_metainfo=None,
    lattice_metainfo=None,
):
    scale = 0.5
    return traj_clothoid[:, -1] * scale


def get_similarity_cost(
    traj,
    traj_clothoid,
    opp_poses=None,
    ego_pose=None,
    prev_traj=None,
    dt=None,
    map_metainfo=None,
    lattice_metainfo=None,
):
    """
    the stored prev_traj is in local frame
    """
    scale = 0.25

    # first iteration: prev trajectory initialized to zero
    if abs(np.sum(prev_traj)) < 1e-6:
        return np.zeros((len(traj)))

    local_traj = traj_global2local(ego_pose, traj[..., :2])
    diff = local_traj - prev_traj
    cost = diff * diff
    cost = np.sum(cost, axis=(1, 2)) * scale
    return cost


@njit(cache=True)
def get_map_collision(
    traj,
    traj_clothoid,
    opp_poses=None,
    ego_pose=None,
    prev_traj=None,
    dt=None,
    map_metainfo=None,
    collision_thres=0.4,
):
    if dt is None:
        raise ValueError(
            "Map Distance Transform dt has to be set when using this cost function."
        )
    # points: (n, 2)
    all_traj_pts = np.ascontiguousarray(traj).reshape(-1, 5)  # (nxm, 5)
    collisions = map_collision(
        all_traj_pts[:, 0:2], dt, map_metainfo, eps=collision_thres
    )  # (nxm)
    collisions = collisions.reshape(len(traj), -1)  # (n, m)
    cost = []
    for traj_collision in collisions:
        if np.any(traj_collision):
            cost.append(3000.0)
        else:
            cost.append(0.0)
    return np.array(cost)


# @njit(cache=True)
def get_obstacle_collision_with_v(
    traj, traj_clothoid, v_lattice, opp_poses, prev_oppo_pose, dt=None
):
    max_cost = 10.0
    min_cost = 10.0
    width, length = 0.31, 0.58
    n, m, _ = traj.shape
    k = v_lattice.shape[1]
    cost = np.zeros((n, 1))

    if len(opp_poses) == 0:
        return cost

    traj_xyt = traj[:, :, :3]
    for i, tr in enumerate(traj_xyt):
        # print(i)
        close_p_idx = x2y_distances_argmin(
            np.ascontiguousarray(opp_poses[:, :2]), np.ascontiguousarray(tr[:, :2])
        )
        for opp_pose, p_idx in zip(opp_poses, close_p_idx):
            opp_box = get_vertices(opp_pose[:3], length, width)
            p_box = get_vertices(tr[int(p_idx)], length, width)
            if collision(opp_box, p_box):
                cost[i] = max_cost - p_idx * (max_cost - min_cost) / m
    if np.sum(prev_oppo_pose) == 0:
        cost = np.repeat(cost, k).reshape(n, k)
        return cost
    else:
        cost = np.repeat(cost, k).reshape(n, k)
        # calculate opp pose, assume only one opponent
        oppo_pose = opp_poses[0][:2]
        prev_opp_pose = prev_oppo_pose[0]
        opp_v = (oppo_pose - prev_opp_pose) / float(dt)  # (2, 1)
        # print(opp_v)
        traj_heading = traj[:, -1, 3]  # (n, )
        traj_heading_vec = np.vstack(
            (np.cos(traj_heading), np.sin(traj_heading))
        ).T  # (n, 2)
        opp_v_proj = np.dot(traj_heading_vec, opp_v.reshape(2, 1))  # (n, 1)
        opp_v_proj = np.repeat(opp_v_proj, k).reshape(n, k)  # (n, k)
        v_diff = v_lattice - opp_v_proj
        cost = cost * v_diff
        for i in range(n):
            for j in range(k):
                if cost[i][j] < 0:
                    cost[i][j] = 1.0
                elif cost[i][j] < 1.2:
                    cost[i][j] = 1.2
    return cost


if __name__ == "__main__":
    import gym_envs
    import gymnasium as gym
    from gym_envs.multi_agent_env.planners.planner import run_planner

    render = True
    track_name = "General1"
    track = Track.from_track_name(track_name)

    pp = PurePursuitPlanner(track, params={"vgain": 0.8}, agent_id="npc0")
    env = gym.make(
        "f110-multi-agent-v0",
        track_name=track_name,
        npc_planners=[pp],
        render_mode="human",
    )

    planner = LatticePlanner(track=env.track)

    env.add_render_callback(planner.render_waypoints)

    for _ in range(5):
        run_planner(
            env, planner, render=render, max_steps=500, reset_mode="random_random"
        )

    env.close()
