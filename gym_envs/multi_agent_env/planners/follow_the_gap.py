import numpy as np
from numba import njit
from queue import PriorityQueue

from gym_envs.multi_agent_env.common.track import Track
from gym_envs.multi_agent_env.planners.planner import Planner


@njit(fastmath=False, cache=True)
def preprocess_scan(ranges, danger_thres, rb):
    # mean filter
    proc_ranges = []
    window_size = 10
    for i in range(0, len(ranges), window_size):
        cur_mean = sum(ranges[i : i + window_size]) / window_size
        for _ in range(window_size):
            proc_ranges.append(cur_mean)
    proc_ranges = np.array(proc_ranges)

    # set danger range and ranges too far to zero
    p, n = 0, len(proc_ranges)
    while p < n:
        if proc_ranges[p] <= danger_thres:
            ranges[max(0, p - rb) : p + rb] = 0
            p += rb
        else:
            p += 1
    return proc_ranges


def find_target_point(ranges, safe_thres, max_gap_length=100, min_gap_length=50):
    """_summary_
        Find all the gaps exceed a safe thres.
        Among those qualified gaps, chose the one with a farmost point, calculate the target as the middle of the gap.
    Args:
        ranges (_type_): _description_
        safe_thres (_type_): _description_
        max_gap_length (int, optional): _description_. Defaults to 350.
    Returns:
        target: int
    """
    n = len(ranges)
    safe_p_left, safe_p_right = 0, n - 1
    p = safe_p_left
    safe_range = PriorityQueue()
    while p < n - 1:
        if ranges[p] >= safe_thres:
            safe_p_left = p
            p += 1
            # while p < end_i and ranges[p] >= self.safe_thres and p-safe_p_left <= 290:
            while (
                p < n - 1
                and ranges[p] >= safe_thres
                and p - safe_p_left <= max_gap_length
            ):
                p += 1
            safe_p_right = p - 1
            if safe_p_right != safe_p_left:
                safe_range.put(
                    (
                        -(np.max(ranges[safe_p_left:safe_p_right])),
                        (safe_p_left, safe_p_right),
                    )
                )
        else:
            p += 1
    if safe_range.empty():
        print("no safe range")
        return np.argmax(ranges)
    else:
        while not safe_range.empty():
            safe_p_left, safe_p_right = safe_range.get()[1]
            if safe_p_right - safe_p_left > min_gap_length:
                target = (safe_p_left + safe_p_right) // 2
                if 190 < target < 900:
                    return target


class FollowTheGap(Planner):
    def __init__(self, params: dict = {}, agent_id: str = "ego"):
        super().__init__(agent_id=agent_id)

        self.params = {
            "angle_min": -2.35,
            "scan_size": 1080,
            "safe_threshold": 2.0,
            "danger_threshold": 0.8,
            "rb": 10,
            "P": 0.6,
            "V_scale": 0.1,
            "max_speed": 8.0,
            "vgain": 1.0,
        }
        self.params.update(params)

    def plan(self, observation, agent_id: str = "ego"):
        assert "scan" in observation or (
            self.agent_id in observation and "scan" in observation[self.agent_id]
        ), "expected pose in observation or multiagent observation"
        assert (
            agent_id == self.agent_id
        ), "agent_id must be the same as the planner's agent_id"

        if "scan" in observation:
            observation = {"ego": observation}

        angle_min, scan_size = self.params["angle_min"], self.params["scan_size"]
        angle_increment = 2.0 * abs(angle_min) / scan_size
        safe_threshold, danger_threshold = (
            self.params["safe_threshold"],
            self.params["danger_threshold"],
        )
        rb = self.params["rb"]

        scan = observation[self.agent_id]["scan"]
        filted_scan = preprocess_scan(scan, danger_threshold, rb)
        best_p_idx = find_target_point(filted_scan, safe_threshold)

        steering = (angle_min + best_p_idx * angle_increment) * self.params["P"]
        velocity = (
            self.params["vgain"] * self.params["max_speed"]
            - abs(steering) / self.params["V_scale"]
        )
        velocity = np.clip(velocity, 0, self.params["max_speed"])

        return {"steering": steering, "velocity": velocity}


if __name__ == "__main__":
    import gymnasium
    import gym_envs
    from gym_envs.multi_agent_env.planners.planner import run_planner
    from gym_envs.multi_agent_env.planners.pure_pursuit import PurePursuitPlanner

    np.random.seed(2)
    track_name = "General1"
    track = Track.from_track_name(track_name)

    pp = PurePursuitPlanner(track, params={"vgain": 0.8}, agent_id="npc0")
    env = gymnasium.make(
        "f110-multi-agent-v0",
        track_name=track_name,
        npc_planners=[pp],
        render_mode="human",
    )

    planner = FollowTheGap()

    for _ in range(5):
        run_planner(env, planner, render=True, max_steps=500, reset_mode="random_back")

    env.close()
