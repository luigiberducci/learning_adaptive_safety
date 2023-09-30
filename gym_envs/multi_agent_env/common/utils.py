"""
Planner Helpers
"""
import math

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,), dtype=np.float32)
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],), dtype=np.float32)
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@njit(fastmath=True, cache=True)
def intersect_point(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :] + np.float32(1e-4)
        end = trajectory[i + 1, :] + np.float32(1e-4)
        V = np.ascontiguousarray(end - start).astype(np.float32)

        a = np.dot(V, V)
        b = np.array(2.0, dtype=np.float32) * np.dot(V, start - point)
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - 2.0 * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = (end - start).astype(np.float32)

            a = np.dot(V, V)
            b = np.array(2.0, dtype=np.float32) * np.dot(V, start - point)
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - 2.0 * np.dot(start, point)
                - radius * radius
            )
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(cache=True)
def simple_norm_axis1(vector):
    return np.sqrt(vector[:, 0] ** 2 + vector[:, 1] ** 2)


@njit(cache=True)
def get_wp_xyv_with_interp(L, curr_pos, theta, waypoints, wpNum, interpScale):
    traj_distances = simple_norm_axis1(waypoints[:, :2] - curr_pos)
    nearest_idx = np.argmin(traj_distances)
    nearest_dist = traj_distances[nearest_idx]
    segment_end = nearest_idx
    # count = 0
    if wpNum < 100 and traj_distances[wpNum - 1] < L:
        segment_end = wpNum - 1
    #     # print(traj_distances[-1])
    else:
        while traj_distances[segment_end] < L:
            segment_end = (segment_end + 1) % wpNum
    #     count += 1
    #     if count > wpNum:
    #         segment_end = wpNum - 1
    #         break
    segment_begin = (segment_end - 1 + wpNum) % wpNum
    x_array = np.linspace(
        waypoints[segment_begin, 0], waypoints[segment_end, 0], interpScale
    )
    y_array = np.linspace(
        waypoints[segment_begin, 1], waypoints[segment_end, 1], interpScale
    )
    v_array = np.linspace(
        waypoints[segment_begin, 2], waypoints[segment_end, 2], interpScale
    )
    xy_interp = np.vstack((x_array, y_array)).T
    dist_interp = simple_norm_axis1(xy_interp - curr_pos) - L
    i_interp = np.argmin(np.abs(dist_interp))
    target_global = np.array((x_array[i_interp], y_array[i_interp]))
    new_L = np.linalg.norm(curr_pos - target_global)
    return (
        np.array((x_array[i_interp], y_array[i_interp], v_array[i_interp])),
        new_L,
        nearest_dist,
    )


@njit(fastmath=True, cache=True)
def cartesian_to_frenet(point: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    a, b, c, i = nearest_point(point, trajectory)
    ia, ib = (i - 1) % trajectory.shape[0], (i + 1) % trajectory.shape[0]

    # calculate s
    point_a, point_b = trajectory[ia, :], trajectory[ib, :]
    t = (point[0] - point_a[0]) * (point_b[0] - point_a[0]) + (
        point[1] - point_a[1]
    ) * (point_b[1] - point_a[1])
    t /= (point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2

    if i > 0:
        # to avoid issue when closest wp i indexed as 0
        diffs = trajectory[1:i, :] - trajectory[: i - 1, :]
        s = sum(np.sqrt(np.sum(diffs**2, -1)))
    else:
        s = 0.0

    s += t * np.sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)

    d = distance_point_to_line(point, trajectory[ia], trajectory[ib])

    return np.array((s, d))


@njit(fastmath=False, cache=True)
def distance_point_to_line(
    point: np.ndarray, point_a: np.ndarray, point_b: np.ndarray
) -> float:
    """
    compute distance of point to line passing through a, b

    ref: http://paulbourke.net/geometry/pointlineplane/

    :param point: point from which we want to compute distance to line
    :param point_a: first point in line
    :param point_b: second point in line
    :return: distance
    """

    px, py = (
        point_b[0] - point_a[0],
        point_b[1] - point_a[1],
    )
    norm = px**2 + py**2

    u = ((point[0] - point_a[0]) * px + (point[1] - point_a[1]) * py) / norm
    u = min(max(u, 0), 1)
    sign = px * (point[1] - point_a[1]) - py * (point[0] - point_a[0]) > 0

    xl = point_a[0] + u * px
    yl = point_a[1] + u * py

    dx = xl - point[0]
    dy = yl - point[1]

    d = np.sqrt(dx**2 + dy**2)
    d = d if sign else -d

    return d


"""
LQR utilities
"""


@njit(cache=True)
def solve_lqr(A, B, Q, R, tolerance, max_num_iteration):
    """
    Iteratively calculating feedback matrix K
    Args:
        A: matrix_a
        B: matrix_b
        Q: matrix_q
        R: matrix_r_
        tolerance: lqr_eps
        max_num_iteration: max_iteration
    Returns:
        K: feedback matrix
    """

    M = np.zeros((Q.shape[0], R.shape[1]))

    AT = A.T
    BT = B.T
    MT = M.T

    P = Q
    num_iteration = 0
    diff = math.inf

    while num_iteration < max_num_iteration and diff > tolerance:
        num_iteration += 1
        P_next = (
            AT @ P @ A
            - (AT @ P @ B + M) @ np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT)
            + Q
        )

        # check the difference between P and P_next
        diff = np.abs(np.max(P_next - P))
        P = P_next

    K = np.linalg.pinv(BT @ P @ B + R) @ (BT @ P @ A + MT)

    return K


@njit(cache=True)
def update_matrix(vehicle_state, state_size, timestep, wheelbase):
    """
    calc A and b matrices of linearized, discrete system.
    Args:
        vehicle_state:
        state_size:
        timestep:
        wheelbase:
    Returns:
        A:
        b:
    """

    # Current vehicle velocity
    v = vehicle_state[3]

    # Initialization of the time discrete A matrix
    matrix_ad_ = np.zeros((state_size, state_size))

    matrix_ad_[0][0] = 1.0
    matrix_ad_[0][1] = timestep
    matrix_ad_[1][2] = v
    matrix_ad_[2][2] = 1.0
    matrix_ad_[2][3] = timestep

    # b = [0.0, 0.0, 0.0, v / L].T
    matrix_bd_ = np.zeros((state_size, 1))  # time discrete b matrix
    matrix_bd_[3][0] = v / wheelbase

    return matrix_ad_, matrix_bd_


@njit(cache=True)
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


"""
Helpers used in adaptive pure pursuit and lattice planner
"""


@njit(cache=True)
def get_adaptive_lookahead(velocity, minL, maxL, Lscale):
    return velocity * (maxL - minL) / Lscale + minL


@njit(cache=True)
def get_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.ascontiguousarray(np.array([[c, -s], [s, c]])).astype(np.float32)


@njit(cache=True)
def x2y_distances_argmin(X, Y):
    """
    X: (n, 2)
    Y: (m, 2)

    return (n, 1)
    """
    # pass
    n = len(X)
    min_idx = np.zeros(n)
    for i in range(n):
        diff = Y - X[i]  # (m, 2)
        # It is because numba does not support 'axis' keyword
        norm2 = diff * diff  # (m, 2)
        norm2 = norm2[:, 0] + norm2[:, 1]
        min_idx[i] = np.argmin(norm2)
    return min_idx


@njit(cache=True)
def get_vertices(pose, length, width):
    """
    Utility function to return vertices of the car body given pose and size
    Args:
        pose (np.ndarray, (3, )): current world coordinate pose of the vehicle
        length (float): car length
        width (float): car width
    Returns:
        vertices (np.ndarray, (4, 2)): corner vertices of the vehicle body
    """
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    x, y = pose[0], pose[1]
    tl_x = -length / 2 * c + width / 2 * (-s) + x
    tl_y = -length / 2 * s + width / 2 * c + y
    tr_x = length / 2 * c + width / 2 * (-s) + x
    tr_y = length / 2 * s + width / 2 * c + y
    bl_x = -length / 2 * c + (-width / 2) * (-s) + x
    bl_y = -length / 2 * s + (-width / 2) * c + y
    br_x = length / 2 * c + (-width / 2) * (-s) + x
    br_y = length / 2 * s + (-width / 2) * c + y
    vertices = np.asarray([[tl_x, tl_y], [bl_x, bl_y], [br_x, br_y], [tr_x, tr_y]])
    # assert np.linalg.norm(vertices_1-vertices) < 1e-4
    # print(vertices_1, vertices)
    return vertices


@njit(cache=True)
def map_collision(points, dt, map_metainfo, eps=0.4):
    """
    Check wheter a point is in collision with the map

    Args:
        points (numpy.ndarray(N, 2)): points to check
        dt (numpy.ndarray(n, m)): the map distance transform
        map_metainfo (tuple (x, y, c, s, h, w, resol)): map metainfo
        eps (float, default=0.1): collision threshold
    Returns:
        collisions (numpy.ndarray (N, )): boolean vector of wheter input points are in collision

    """
    orig_x, orig_y, orig_c, orig_s, height, width, resolution, tracklen = map_metainfo
    collisions = np.empty((points.shape[0],))
    for i in range(points.shape[0]):
        r, c = xy_2_rc(
            points[i, 0],
            points[i, 1],
            orig_x,
            orig_y,
            orig_c,
            orig_s,
            height,
            width,
            resolution,
        )
        if dt[r, c] <= eps:
            collisions[i] = True
        else:
            collisions[i] = False
    return np.ascontiguousarray(collisions)


@njit(cache=True)
def xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution):
    """
    Translate (x, y) coordinate into (r, c) in the matrix
        Args:
            x (float): coordinate in x (m)
            y (float): coordinate in y (m)
            orig_x (float): x coordinate of the map origin (m)
            orig_y (float): y coordinate of the map origin (m)

        Returns:
            r (int): row number in the transform matrix of the given point
            c (int): column number in the transform matrix of the given point
    """
    # translation
    x_trans = x - orig_x
    y_trans = y - orig_y

    # rotation
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c

    # clip the state to be a cell
    if (
        x_rot < 0
        or x_rot >= width * resolution
        or y_rot < 0
        or y_rot >= height * resolution
    ):
        c = -1
        r = -1
    else:
        c = int(x_rot / resolution)
        r = int(y_rot / resolution)

    return r, c


@njit(cache=True)
def collision(vertices1, vertices2):
    """
    GJK test to see whether two bodies overlap
    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body
    Returns:
        overlap (boolean): True if two bodies collide
    """
    index = 0
    simplex = np.empty((3, 2))

    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    d = position1 - position2

    if d[0] == 0 and d[1] == 0:
        d[0] = 1.0

    a = support(vertices1, vertices2, d)
    simplex[index, :] = a

    if d.dot(a) <= 0:
        return False

    d = -a

    iter_count = 0
    while iter_count < 1e3:
        a = support(vertices1, vertices2, d)
        index += 1
        simplex[index, :] = a
        if d.dot(a) <= 0:
            return False

        ao = -a

        if index < 2:
            b = simplex[0, :]
            ab = b - a
            d = tripleProduct(ab, ao, ab)
            if np.linalg.norm(d) < 1e-10:
                d = perpendicular(ab)
            continue

        b = simplex[1, :]
        c = simplex[0, :]
        ab = b - a
        ac = c - a

        acperp = tripleProduct(ab, ac, ac)

        if acperp.dot(ao) >= 0:
            d = acperp
        else:
            abperp = tripleProduct(ac, ab, ab)
            if abperp.dot(ao) < 0:
                return True
            simplex[0, :] = simplex[1, :]
            d = abperp

        simplex[1, :] = simplex[2, :]
        index -= 1

        iter_count += 1
    return False


@njit(cache=True)
def tripleProduct(a, b, c):
    """
    Return triple product of three vectors
    Args:
        a, b, c (np.ndarray, (2,)): input vectors
    Returns:
        (np.ndarray, (2,)): triple product
    """
    ac = a.dot(c)
    bc = b.dot(c)
    return b * ac - a * bc


@njit(cache=True)
def perpendicular(pt):
    """
    Return a 2-vector's perpendicular vector
    Args:
        pt (np.ndarray, (2,)): input vector
    Returns:
        pt (np.ndarray, (2,)): perpendicular vector
    """
    temp = pt[0]
    pt[0] = pt[1]
    pt[1] = -1 * temp
    return pt


@njit(cache=True)
def support(vertices1, vertices2, d):
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]


@njit(cache=True)
def avgPoint(vertices):
    return np.sum(vertices, axis=0) / vertices.shape[0]


@njit(cache=True)
def indexOfFurthestPoint(vertices, d):
    return np.argmax(vertices.dot(d))


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


@njit(cache=True)
def zero_2_2pi(angle):
    if angle > 2 * np.pi:
        return angle - np.float32(2.0 * math.pi)
    if angle < 0:
        return angle + np.float32(2.0 * math.pi)

    return angle


@njit(cache=True)
def cal_ittc(poses_1, poses_2, D, length=0.58, width=0.31, dt=0.01):
    n = poses_1.shape[0]
    all_dist = np.empty(n)
    for i, pose_1, pose_2, direct in zip(range(n), poses_1, poses_2, D):
        vertice_1 = get_vertices(pose_1, length, width)
        vertice_2 = get_vertices(pose_2, length, width)

        if collision(vertice_1, vertice_2):
            break

        dist = distance(vertice_1, vertice_2, direct)
        all_dist[i] = dist
    all_dist = all_dist[:i]
    delta_d = -(all_dist[1:] - all_dist[:-1])
    delta_d = np.clip(delta_d, a_min=1e-3, a_max=100.0)
    ittc = all_dist[:-1] * dt / delta_d
    return ittc


@njit(cache=True)
def distance(vertices1, vertices2, direc):
    vertices1 = np.ascontiguousarray(vertices1)
    vertices2 = np.ascontiguousarray(vertices2)
    direc = np.ascontiguousarray(direc)
    a = support(vertices1, vertices2, direc)

    b = support(vertices1, vertices2, -direc)
    d = closestPoint2Origin(a, b)
    dist = np.linalg.norm(d)
    while True:
        if dist < 1e-10:
            return dist
        d = -d
        c = support(vertices1, vertices2, d)
        temp1 = c.dot(d)
        temp2 = a.dot(d)
        # should get bigger along d or you hit the end
        if (temp1 - temp2) < 1e-10:
            return dist
        p1 = closestPoint2Origin(a, c)
        p2 = closestPoint2Origin(c, b)
        dist1 = np.linalg.norm(p1)
        dist2 = np.linalg.norm(p2)
        if dist1 < dist2:
            b = c
            d = p1
            dist = dist1
        else:
            a = c
            d = p2
            dist = dist2


@njit(cache=True)
def closestPoint2Origin(a, b):
    ab = b - a
    ao = -a
    length = ab.dot(ab)
    if length < 1e-10:
        return a
    frac = ao.dot(ab) / length
    if frac < 0:
        return a
    if frac > 1:
        return b
    return frac * ab + a


@njit(cache=True)
def S_Plus(ori, plus, mod):
    """
    in a circle, assume the ori should be ahead
    """
    return (ori + mod + plus) % mod
