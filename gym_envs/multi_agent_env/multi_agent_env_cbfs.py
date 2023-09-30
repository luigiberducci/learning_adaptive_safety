import casadi
import gymnasium as gym
import numpy as np

debug = False

cbf_params = {
    "acceleration_range": [-9.51, 9.51],
    "beta_range": [-0.22, 0.22],
    "lf": 0.15875,
    "lr": 0.17145,
    "dt": 0.1,
    "track_width": 3.00,
    "wall_margin": 0.58,
    "safe_distance": 0.58 + 0.2,
    # additional params, used only in opt-decay cbf (odcbf)
    "odcbf_gamma_range": [0.0, 1.0],  # range of gamma, for optimal-decay cbf
    "odcbf_gamma_penalty": 1e4,  # penalty for deviation from nominal gamma, for optimal-decay cbf
}


# cbf utils
def hx_left_wall(ey, dey, track_halfw, max_brake_lat, safety_margin):
    dist_left = track_halfw - ey
    Dv_left = -dey  # negative if moving left
    Dv_left = casadi.fmin(Dv_left, 0.0)
    hx_left = dist_left - (Dv_left**2) / (2 * max_brake_lat) - safety_margin
    return hx_left


def hx_right_wall(ey, dey, track_halfw, max_brake_lat, safety_margin):
    dist_right = track_halfw + ey
    Dv_right = dey  # negative if moving right
    Dv_right = casadi.fmin(Dv_right, 0.0)
    hx_right = dist_right - (Dv_right**2) / (2 * max_brake_lat) - safety_margin
    return hx_right


def hx_obstacle(rel_pos, rel_vel, max_brake_long, safety_distance):
    norm_pos = casadi.norm_2(rel_pos)
    Dp = rel_pos / norm_pos

    Dv = casadi.dot(rel_vel, Dp)
    Dv = casadi.fmin(Dv, 0.0)

    hx_obs = norm_pos - (Dv**2) / (2 * max_brake_long) - safety_distance
    return hx_obs


def f1tenth_cbf_project(env, state, action, gammas):
    params = cbf_params

    # optimizer
    opti = casadi.Opti()  # casadi.Opti("conic")

    # decision variables
    a_min, a_max = params["acceleration_range"]
    beta_min, beta_max = params["beta_range"]
    u_min = np.array([a_min, beta_min])
    u_max = np.array([a_max, beta_max])

    u = opti.variable(2)
    opti.subject_to(u_min <= u)
    opti.subject_to(u <= u_max)

    # extract state variables
    x, y, psi = state["ego"]["pose"]
    v = state["ego"]["velocity"][0]
    s, ey = state["ego"]["frenet_coords"]
    s = s % env.track.centerline.ss[-1]
    dt = params["dt"]
    lf, lr = params["lf"], params["lr"]
    wb = lf + lr

    # convert action (vx, steering angle) to input (accx, beta)
    delta, target_v = action
    accx = (target_v - v) / dt
    accx = np.clip(accx, a_min, a_max)

    beta = np.arctan(lr / wb * np.tan(delta))
    beta = np.clip(beta, beta_min, beta_max)

    # opponent states
    obstacle_states = []
    for obstacle_id in env.agents_ids:
        if obstacle_id == "ego":
            continue
        ox, oy, opsi = state[obstacle_id]["pose"]
        ov = state[obstacle_id]["velocity"][0]
        os, oey = state[obstacle_id]["frenet_coords"]
        os = os % env.track.centerline.ss[-1]

        if -0.3 < os - s < 5.0:
            obstacle_states.append([ox, oy, os, oey, opsi, ov])

    # extract track info
    k_s = env.track.centerline.spline.calc_curvature(s=s)
    psi_s = env.track.centerline.spline.calc_yaw(s=s)

    # slacks
    wall_slack = opti.variable(2)

    if len(obstacle_states) > 0:
        # obstacle slacks
        obstacle_slacks = opti.variable(len(obstacle_states))

    # next state variables
    new_x = x + dt * (v * np.cos(psi) - v * np.sin(psi) * u[1])
    new_y = y + dt * (v * np.sin(psi) + v * np.cos(psi) * u[1])

    denom = 1 - ey * k_s
    new_s = s + dt * (
        v * np.cos(psi - psi_s) / denom - v * np.sin(psi - psi_s) / denom * u[1]
    )
    new_ey = ey + dt * (v * np.sin(psi - psi_s) + v * np.cos(psi - psi_s) * u[1])

    new_v = v + dt * u[0]
    new_psi = psi + dt * (v / lr) * u[1]

    # braking accelerations lat and long
    max_brake_lat = new_max_brake_lat = 1.0  # fixed
    max_brake_long = 1.0 - (v / 8.0)  # velocity dependent
    new_max_brake_long = 1.0 - (new_v / 8.0)  # velocity dependent

    # wall constraint
    track_halfw, safety_margin = params["track_width"] / 2, params["wall_margin"]

    dey = v * np.sin(psi - psi_s) + v * np.cos(psi - psi_s) * u[1]
    new_dey = new_v * np.sin(psi - psi_s) + v * np.cos(psi - psi_s) * u[1]

    hx_left = hx_left_wall(ey, dey, track_halfw, max_brake_lat, safety_margin)
    hx_left_next = hx_left_wall(
        new_ey, new_dey, track_halfw, new_max_brake_lat, safety_margin
    )

    opti.subject_to(hx_left_next - hx_left + wall_slack[0] >= -gammas[0] * hx_left)
    opti.subject_to(wall_slack[0] >= 0)

    hx_right = hx_right_wall(ey, dey, track_halfw, max_brake_lat, safety_margin)
    hx_right_next = hx_right_wall(
        new_ey, new_dey, track_halfw, new_max_brake_lat, safety_margin
    )

    opti.subject_to(hx_right_next - hx_right + wall_slack[1] >= -gammas[0] * hx_right)
    opti.subject_to(wall_slack[1] >= 0)

    # obstacle constraints
    hx_obss = []
    for i, obstacle_state in enumerate(obstacle_states):
        obs_x, obs_y, obs_s, obs_ey, obs_psi, obs_v = obstacle_state
        obs_beta = 0.0

        next_obs_x = obs_x + dt * (
            obs_v * np.cos(obs_psi) - obs_v * np.sin(obs_psi) * obs_beta
        )
        next_obs_y = obs_y + dt * (
            obs_v * np.sin(obs_psi) + obs_v * np.cos(obs_psi) * obs_beta
        )
        next_obs_psi = obs_psi + dt * (obs_v / lr) * obs_beta

        rel_pos = casadi.vertcat(obs_x - x, obs_y - y)
        ego_vx, ego_vy = v * np.cos(psi), v * np.sin(psi)
        obs_vx, obs_vy = obs_v * np.cos(obs_psi), obs_v * np.sin(obs_psi)
        rel_vel = casadi.vertcat(obs_vx - ego_vx, obs_vy - ego_vy)

        new_rel_pos = casadi.vertcat(next_obs_x - new_x, next_obs_y - new_y)
        new_ego_vx, new_ego_vy = new_v * np.cos(new_psi), new_v * np.sin(new_psi)
        new_obs_vx, new_obs_vy = (
            obs_v * np.cos(next_obs_psi),
            obs_v * np.sin(next_obs_psi),
        )
        new_rel_vel = casadi.vertcat(new_obs_vx - new_ego_vx, new_obs_vy - new_ego_vy)

        # hx
        hx_obs = hx_obstacle(rel_pos, rel_vel, max_brake_long, params["safe_distance"])
        hx_obs_next = hx_obstacle(
            new_rel_pos, new_rel_vel, new_max_brake_long, params["safe_distance"]
        )

        opti.subject_to(
            hx_obs_next - hx_obs + obstacle_slacks[i] >= -gammas[1] * hx_obs
        )
        opti.subject_to(obstacle_slacks[i] >= 0)

        # for logging
        hx_obss.append(float(hx_obs))

    # objective
    a_scale, beta_scale = 1.0 / a_max, 1.0 / beta_max
    obj = (
        a_scale * (u[0] - accx) ** 2
        + beta_scale * (u[1] - beta) ** 2
        + 1000 * casadi.sum1(wall_slack)
    )
    if len(obstacle_states) > 0:
        obj += 1000 * casadi.sum1(obstacle_slacks)

    opti.minimize(obj)

    # solve
    p_opts = {"print_time": False, "verbose": False}
    s_opts = {"print_level": 0}
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        safe_input = sol.value(u)

        dict_infos = {
            "hx_left": sol.value(hx_left),
            "hx_right": sol.value(hx_right),
            "slack_left": sol.value(wall_slack[0]),
            "slack_right": sol.value(wall_slack[1]),
            "obj": sol.value(obj),
            "gammas": sol.value(gammas),
            "safe_gammas": sol.value(gammas),
            "opt_status": 1.0,
        }
        for i in range(len(obstacle_states)):
            dict_infos[f"hx_obs_{i}"] = hx_obss[i]
            dict_infos[f"slack_obs_{i}"] = sol.value(obstacle_slacks[i])

    except Exception:
        safe_input = accx, beta

        dict_infos = {
            "hx_left": 0.0,
            "hx_right": 0.0,
            "slack_left": 0.0,
            "slack_right": 0.0,
            "obj": 0.0,
            "gammas": gammas,
            "safe_gammas": gammas,
            "opt_status": -1.0,
        }
        for i in range(len(obstacle_states)):
            dict_infos[f"hx_obs_{i}"] = 0.0
            dict_infos[f"slack_obs_{i}"] = 0.0

    # convert safe input to action (vx, steering angle)
    safe_accx, safe_beta = safe_input
    target_v = v + dt * safe_accx
    delta = np.arctan(wb / lr * np.tan(safe_beta))
    safe_action = np.array([delta, target_v])

    if debug:
        for k, v in dict_infos.items():
            print(k, v)

    return safe_action, dict_infos


def f1tenth_optdecay_cbf_project(env, state, action, gammas):
    params = cbf_params

    # optimizer
    opti = casadi.Opti()  # casadi.Opti("conic")

    # decision variables
    a_min, a_max = params["acceleration_range"]
    beta_min, beta_max = params["beta_range"]
    u_min = np.array([a_min, beta_min])
    u_max = np.array([a_max, beta_max])

    u = opti.variable(2)
    opti.subject_to(u_min <= u)
    opti.subject_to(u <= u_max)

    # optimal-decay formulation
    optimal_gammas = opti.variable(2)

    gamma_min, gamma_max = params["odcbf_gamma_range"]
    opti.subject_to(gamma_min <= optimal_gammas)
    opti.subject_to(optimal_gammas <= gamma_max)

    # extract state variables
    x, y, psi = state["ego"]["pose"]
    v = state["ego"]["velocity"][0]
    s, ey = state["ego"]["frenet_coords"]
    s = s % env.track.centerline.ss[-1]
    dt = params["dt"]
    lf, lr = params["lf"], params["lr"]
    wb = lf + lr

    # convert action (vx, steering angle) to input (accx, beta)
    delta, target_v = action
    accx = (target_v - v) / dt
    accx = np.clip(accx, a_min, a_max)

    beta = np.arctan(lr / wb * np.tan(delta))
    beta = np.clip(beta, beta_min, beta_max)

    # opponent states
    obstacle_states = []
    for obstacle_id in env.agents_ids:
        if obstacle_id == "ego":
            continue
        ox, oy, opsi = state[obstacle_id]["pose"]
        ov = state[obstacle_id]["velocity"][0]
        os, oey = state[obstacle_id]["frenet_coords"]
        os = os % env.track.centerline.ss[-1]

        if -0.3 < os - s < 5.0:
            obstacle_states.append([ox, oy, os, oey, opsi, ov])

    # extract track info
    k_s = env.track.centerline.spline.calc_curvature(s=s)
    psi_s = env.track.centerline.spline.calc_yaw(s=s)

    # slacks
    wall_slack = opti.variable(2)

    if len(obstacle_states) > 0:
        # obstacle slacks
        obstacle_slacks = opti.variable(len(obstacle_states))

    # next state variables
    new_x = x + dt * (v * np.cos(psi) - v * np.sin(psi) * u[1])
    new_y = y + dt * (v * np.sin(psi) + v * np.cos(psi) * u[1])

    denom = 1 - ey * k_s
    new_s = s + dt * (
        v * np.cos(psi - psi_s) / denom - v * np.sin(psi - psi_s) / denom * u[1]
    )
    new_ey = ey + dt * (v * np.sin(psi - psi_s) + v * np.cos(psi - psi_s) * u[1])

    new_v = v + dt * u[0]
    new_psi = psi + dt * (v / lr) * u[1]

    # braking accelerations lat and long
    max_brake_lat = new_max_brake_lat = 1.0  # fixed
    max_brake_long = 1.0 - (v / 8.0)  # velocity-dependent
    new_max_brake_long = 1.0 - (new_v / 8.0)

    # wall constraint
    track_halfw, safety_margin = params["track_width"] / 2, params["wall_margin"]

    dey = v * np.sin(psi - psi_s) + v * np.cos(psi - psi_s) * u[1]
    new_dey = new_v * np.sin(psi - psi_s) + v * np.cos(psi - psi_s) * u[1]

    hx_left = hx_left_wall(ey, dey, track_halfw, max_brake_lat, safety_margin)
    hx_left_next = hx_left_wall(
        new_ey, new_dey, track_halfw, new_max_brake_lat, safety_margin
    )

    opti.subject_to(
        hx_left_next - hx_left + wall_slack[0] >= -optimal_gammas[0] * hx_left
    )
    opti.subject_to(wall_slack[0] >= 0)

    hx_right = hx_right_wall(ey, dey, track_halfw, max_brake_lat, safety_margin)
    hx_right_next = hx_right_wall(
        new_ey, new_dey, track_halfw, new_max_brake_lat, safety_margin
    )

    opti.subject_to(
        hx_right_next - hx_right + wall_slack[1] >= -optimal_gammas[0] * hx_right
    )
    opti.subject_to(wall_slack[1] >= 0)

    # obstacle constraints
    hx_obss = []
    for i, obstacle_state in enumerate(obstacle_states):
        obs_x, obs_y, obs_s, obs_ey, obs_psi, obs_v = obstacle_state
        obs_beta = 0.0

        next_obs_x = obs_x + dt * (
            obs_v * np.cos(obs_psi) - obs_v * np.sin(obs_psi) * obs_beta
        )
        next_obs_y = obs_y + dt * (
            obs_v * np.sin(obs_psi) + obs_v * np.cos(obs_psi) * obs_beta
        )
        next_obs_psi = obs_psi + dt * (obs_v / lr) * obs_beta

        rel_pos = casadi.vertcat(obs_x - x, obs_y - y)
        ego_vx, ego_vy = v * np.cos(psi), v * np.sin(psi)
        obs_vx, obs_vy = obs_v * np.cos(obs_psi), obs_v * np.sin(obs_psi)
        rel_vel = casadi.vertcat(obs_vx - ego_vx, obs_vy - ego_vy)

        new_rel_pos = casadi.vertcat(next_obs_x - new_x, next_obs_y - new_y)
        new_ego_vx, new_ego_vy = new_v * np.cos(new_psi), new_v * np.sin(new_psi)
        new_obs_vx, new_obs_vy = (
            obs_v * np.cos(next_obs_psi),
            obs_v * np.sin(next_obs_psi),
        )
        new_rel_vel = casadi.vertcat(new_obs_vx - new_ego_vx, new_obs_vy - new_ego_vy)

        # hx
        hx_obs = hx_obstacle(rel_pos, rel_vel, max_brake_long, params["safe_distance"])
        hx_obs_next = hx_obstacle(
            new_rel_pos, new_rel_vel, new_max_brake_long, params["safe_distance"]
        )

        opti.subject_to(
            hx_obs_next - hx_obs + obstacle_slacks[i] >= -optimal_gammas[1] * hx_obs
        )
        opti.subject_to(obstacle_slacks[i] >= 0)

        # for logging
        hx_obss.append(float(hx_obs))

    # objective
    gamma_penalty = cbf_params["odcbf_gamma_penalty"]
    a_scale, beta_scale = 1.0 / a_max, 1.0 / beta_max
    obj = (
        a_scale * (u[0] - accx) ** 2
        + beta_scale * (u[1] - beta) ** 2
        + 1000000 * casadi.sum1(wall_slack)
    )
    if len(obstacle_states) > 0:
        obj += 1000000 * casadi.sum1(obstacle_slacks)
    # add optimal-decay term
    obj += gamma_penalty * casadi.sumsqr(optimal_gammas - gammas)

    opti.minimize(obj)

    # solve
    p_opts = {"print_time": False, "verbose": False}
    s_opts = {"print_level": 0}
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        safe_input = sol.value(u)
        optimal_gammas = sol.value(optimal_gammas)

        dict_infos = {
            "hx_left": sol.value(hx_left),
            "hx_right": sol.value(hx_right),
            "slack_left": sol.value(wall_slack[0]),
            "slack_right": sol.value(wall_slack[1]),
            "obj": sol.value(obj),
            "gammas": sol.value(gammas),
            "optimal_gammas": sol.value(optimal_gammas),
            "opt_status": 1.0,
        }
        for i in range(len(obstacle_states)):
            dict_infos[f"hx_obs_{i}"] = hx_obss[i]
            dict_infos[f"slack_obs_{i}"] = sol.value(obstacle_slacks[i])

    except Exception:
        safe_input = accx, beta

        dict_infos = {
            "hx_left": 0.0,
            "hx_right": 0.0,
            "slack_left": 0.0,
            "slack_right": 0.0,
            "obj": 0.0,
            "gammas": gammas,
            "optimal_gammas": np.zeros_like(gammas),
            "opt_status": -1.0,
        }
        for i in range(len(obstacle_states)):
            dict_infos[f"hx_obs_{i}"] = 0.0
            dict_infos[f"slack_obs_{i}"] = 0.0

    # convert safe input to action (vx, steering angle)
    safe_accx, safe_beta = safe_input
    target_v = v + dt * safe_accx
    delta = np.arctan(wb / lr * np.tan(safe_beta))
    safe_action = np.array([delta, target_v])

    if debug:
        for k, v in dict_infos.items():
            print(k, v)

    return safe_action, dict_infos


def cbf_factory_f1tenth(cbf_type: str, env):
    if cbf_type == "simple":
        return lambda state, action, gamma: f1tenth_cbf_project(
            env, state, action, gamma
        )
    elif cbf_type == "simpledecay":
        return lambda state, action, gamma: f1tenth_optdecay_cbf_project(
            env, state, action, gamma
        )

    else:
        raise ValueError(f"unknown cbf type {cbf_type}")


if __name__ == "__main__":
    import gym_envs
    from functools import partial
    from gym_envs.wrappers.cbf_wrappers import CBFSafetyLayer
    from gym_envs.multi_agent_env.planners.lattice_planner import LatticePlanner
    from gym_envs.multi_agent_env.common.track import Track
    from gym_envs.wrappers.action_wrappers import FlattenAction
    from gym_envs.multi_agent_env.planners.pure_pursuit import PurePursuitPlanner

    # init env
    np.random.seed(1)

    track_name = "General1"
    track = Track.from_track_name(track_name)
    opps = [
        PurePursuitPlanner(
            track=track,
            params={"vgain": 0.7, "tracker": "pure_pursuit"},
            agent_id=f"npc{i}",
        )
        for i in range(3)
    ]
    env = gym.make(
        "f110-multi-agent-v0",
        track_name=track_name,
        params={
            "reset": {"default_reset_mode": "random_back"},
            "simulation": {"control_frequency": 2},
            "termination": {"timeout": 30.0},
        },
        npc_planners=opps,
        render_mode="human",
    )
    env = FlattenAction(env)

    planner = LatticePlanner(
        track=track, params={"vgain": 0.8, "tracker": "pure_pursuit"}, agent_id="ego"
    )

    # rendering
    def precompute_wall_boundaries(centerline, margin=0.0):
        """
        Loop over centerline waypoints and compute the left/right boundaries of the track
        at +-(track_width/2 - safety_margin).

        :param centerline: centerline as raceline object
        :return: list of (x, y) points
        """
        points = []
        dist = cbf_params["track_width"] / 2 - margin
        for s in centerline.ss:
            xc, yc = centerline.spline.calc_position(s=s)
            psi = centerline.spline.calc_yaw(s=s)

            # compute left/right boundaries of the track, projecting along the normal vector at the current point
            norm_vec = np.array([np.cos(psi + np.pi / 2), np.sin(psi + np.pi / 2)])
            left = np.array([xc, yc]) + dist * norm_vec
            right = np.array([xc, yc]) - dist * norm_vec

            points.append(left)
            points.append(right)

        return np.array(points)

    wall_boundaries = precompute_wall_boundaries(env.track.centerline)
    cbf_wall_boundaries = precompute_wall_boundaries(
        env.track.centerline, margin=cbf_params["wall_margin"]
    )
    render_cbf_wall_boundaries, render_cbf_opp_boundaries = [], []

    def render_waypoints(e):
        planner.render_waypoints(e)

        from pyglet.gl import GL_POINTS

        scaled_points = 50.0 * cbf_wall_boundaries

        for i in range(scaled_points.shape[0]):
            if len(render_cbf_wall_boundaries) < scaled_points.shape[0]:
                b = e.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ("v3f/stream", [scaled_points[i, 0], scaled_points[i, 1], 0.0]),
                    ("c3B/stream", (255, 255, 0)),
                )
                render_cbf_wall_boundaries.append(b)
            else:
                render_cbf_wall_boundaries[i].vertices = [
                    scaled_points[i, 0],
                    scaled_points[i, 1],
                    0.0,
                ]

        obst_points = [
            env.state[agent_id]["pose"][:2]
            for agent_id in env.state
            if agent_id != "ego"
        ]
        scaled_points = 50.0 * np.array(obst_points)

        # create points of circle around each scaled point with radius agent_size
        radius = cbf_params["safe_distance"] * 50.0
        circle_pts = np.array(
            [
                np.array([np.cos(theta), np.sin(theta)]) * radius + scaled_points[i]
                for i in range(len(scaled_points))
                for theta in np.linspace(0, 2 * np.pi, 50)
            ]
        )
        for i in range(circle_pts.shape[0]):
            if len(render_cbf_opp_boundaries) < circle_pts.shape[0]:
                b = e.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ("v3f/stream", [circle_pts[i, 0], circle_pts[i, 1], 0.0]),
                    ("c3B/stream", [255, 255, 0]),
                )
                render_cbf_opp_boundaries.append(b)
            else:
                render_cbf_opp_boundaries[i].vertices = [
                    circle_pts[i, 0],
                    circle_pts[i, 1],
                    0.0,
                ]

    # env.add_render_callback(planner.render_waypoints)
    env.add_render_callback(render_waypoints)

    # make_cbf = partial(cbf_factory_f1tenth, cbf_type="simple")
    make_cbf = partial(cbf_factory_f1tenth, cbf_type="simple")

    gamma_range = [0.0, 1.0]
    env = CBFSafetyLayer(
        env, safety_dim=2, alpha=1.0, gamma_range=gamma_range, make_cbf=make_cbf
    )

    obs, _ = env.reset()
    planner.reset()
    done = False

    actions = []
    safe_actions = []
    track_curvature = []
    track_yaw = []
    trajectory_s, trajectory_ey, trajectory_v, d2ego, v2ego = {}, {}, {}, {}, {}
    hxs = {}

    while not done:
        # define nominal action and adaptive safety term gamma
        action = planner.plan(obs)
        action = np.array([action["steering"], action["velocity"]])

        gamma = 0.9
        gamma_actions = -1 + 2 * (
            gamma * np.ones(2)
        )  # do not touch this: normalization gamma to [-1, 1]

        # concatenate action and gamma into a new action
        action = np.concatenate([action, gamma_actions])
        # action_gamma = env.action_space.sample()

        # step the environment
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        # offline analysis
        for agent_id in env.agents_ids:
            if agent_id not in trajectory_s:
                trajectory_s[agent_id] = []
                trajectory_ey[agent_id] = []
                trajectory_v[agent_id] = []
                d2ego[agent_id] = []
                v2ego[agent_id] = []

            s, d = obs[agent_id]["frenet_coords"]
            s = s % env.track.centerline.ss[-1]

            v = obs[agent_id]["velocity"][0]
            ve = obs["ego"]["velocity"][0]

            x, y, _ = obs[agent_id]["pose"]
            xe, ye, _ = obs["ego"]["pose"]

            rel_d = np.linalg.norm([x - xe, y - ye])
            rel_v = v - ve
            d2ego[agent_id].append(rel_d)
            v2ego[agent_id].append(rel_v)

            trajectory_s[agent_id].append(s)
            trajectory_ey[agent_id].append(d)
            trajectory_v[agent_id].append(v)

        k_s = env.track.centerline.spline.calc_smooth_curvature(s=s)
        psi_s = env.track.centerline.spline.calc_smooth_yaw(s=s)

        track_curvature.append(k_s)
        track_yaw.append(psi_s)

        if "safe_action" in info:
            actions.append(info["action"])
            safe_actions.append(info["safe_action"])

            for k, v in info.items():
                if k.startswith("hx"):
                    if k not in hxs:
                        hxs[k] = []
                    hxs[k].append(v)

    if "cbf_stats" in info:
        [
            print(k, v)
            for k, v in info["cbf_stats"].items()
            if k.startswith("episodic_hx")
        ]

    env.close()

    import matplotlib.pyplot as plt

    actions = np.array(actions)
    safe_actions = np.array(safe_actions)

    fig, axes = plt.subplots(5, 2, figsize=(10, 5))

    ax = axes[0, 0]
    ax.plot(actions[:, 0], label="steering")
    ax.plot(safe_actions[:, 0], label="safe steering")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(actions[:, 1], label="velocity")
    ax.plot(safe_actions[:, 1], label="safe velocity")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(track_curvature, label="track curvature")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(track_yaw, label="track yaw")
    ax.legend()

    ax = axes[2, 0]
    for agent_id in trajectory_s:
        ax.plot(trajectory_s[agent_id], label=f"s - {agent_id}")
    ax.legend()

    ax = axes[2, 1]
    tw = cbf_params["track_width"] / 2.0
    tw_sm = tw - cbf_params["wall_margin"]
    ax.hlines(0.0, 0, len(trajectory_ey["ego"]), color="black", linestyle="--")
    ax.hlines(tw, 0, len(trajectory_ey["ego"]), color="black")
    ax.hlines(-tw, 0, len(trajectory_ey["ego"]), color="black")
    ax.hlines(
        tw_sm, 0, len(trajectory_ey["ego"]), color="black", linestyle="-.", label="left"
    )
    ax.hlines(
        -tw_sm,
        0,
        len(trajectory_ey["ego"]),
        color="black",
        linestyle="--",
        label="right",
    )
    for agent_id in trajectory_ey:
        ax.plot(trajectory_ey[agent_id], label=f"ey - {agent_id}")
    ax.legend()

    ax = axes[3, 0]
    for agent_id in trajectory_v:
        ax.plot(v2ego[agent_id], label=f"v2ego - {agent_id}")
        ax.plot(d2ego[agent_id], label=f"d2ego - {agent_id}")
    ax.legend()

    ax = axes[4, 0]
    for k, v in hxs.items():
        ax.plot(v, label=k)
    ax.hlines(0.0, 0, len(v), color="black", linestyles="--")
    ax.legend()

    plt.legend()
    plt.show()
