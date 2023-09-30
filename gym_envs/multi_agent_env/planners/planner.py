from abc import abstractmethod
from typing import Dict

import gymnasium as gym
import numpy as np


class Planner:
    def __init__(self, agent_id: str = "ego"):
        self.agent_id = agent_id
        self.params = {}
        self.waypoints = None
        self.drawn_waypoints = []
        self.render_waypoints_rgb = None

    @abstractmethod
    def plan(self, *args, **kwargs) -> Dict[str, float]:
        pass

    @abstractmethod
    def reset(self, **kwargs):
        pass

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        from pyglet.gl import GL_POINTS

        if self.waypoints is None:
            return

        if self.render_waypoints_rgb is not None:
            rgb = self.render_waypoints_rgb
        else:
            rgb = (183, 193, 222)

        points = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T

        scaled_points = 50.0 * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ("v3f/stream", [scaled_points[i, 0], scaled_points[i, 1], 0.0]),
                    ("c3B/stream", rgb),
                )
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [
                    scaled_points[i, 0],
                    scaled_points[i, 1],
                    0.0,
                ]


class DummyPlanner(Planner):
    def __init__(self, params: dict = {}, agent_id: str = "ego"):
        super().__init__(agent_id=agent_id)
        self.params = {"vgain": 1.0}
        self.params.update(params)

    def plan(self, *args, **kwargs) -> Dict[str, float]:
        return {"steering": 0, "velocity": self.params["vgain"]}

    def reset(self, **kwargs):
        pass


class RandomPlanner(Planner):
    def __init__(
        self,
        min_steering: float = -0.5,
        max_steering: float = 0.5,
        min_velocity: float = 0.0,
        max_velocity: float = 5.0,
        agent_id: str = "ego",
    ):
        super().__init__(agent_id=agent_id)
        self.steering_range = (min_steering, max_steering)
        self.velocity_range = (min_velocity, max_velocity)

    def plan(self, *args, **kwargs) -> Dict[str, float]:
        return {
            "steering": np.random.uniform(*self.steering_range),
            "velocity": np.random.uniform(*self.velocity_range),
        }


class MultiObjectivePlanner(Planner):
    def __init__(self, agent_id: str = "ego"):
        super().__init__(agent_id=agent_id)
        self._safety_target_tradeoff: float = (
            0.5  # default: 0.5 means equal weight to both objectives
        )

    @property
    def safety_target_tradeoff(self) -> float:
        return self._safety_target_tradeoff

    @safety_target_tradeoff.setter
    def safety_target_tradeoff(self, value: float):
        assert 0 <= value <= 1, "Safety/Target tradeoff must be between 0 and 1"
        self._safety_target_tradeoff = value


def run_planner(
    env: gym.Env,
    planner: Planner,
    render: bool = False,
    max_steps: int = np.Inf,
    reset_mode: str = "random_back",
):
    import time

    if reset_mode:
        obs, _ = env.reset(options={"mode": reset_mode})
    else:
        obs, _ = env.reset()

    planner.reset()
    done = False
    steps = 0

    t0 = time.time()
    while not done and steps < max_steps:
        action = planner.plan(obs)
        # print(action)
        obs, reward, done, truncated, info = env.step(action)

        if render:
            env.render()

        steps += 1

    tf = time.time()
    ego_in_front = (
        obs["ego"]["frenet_coords"][0] > obs["npc0"]["frenet_coords"][0]
        if len(env.npc_planners) > 0
        else None
    )
    ego_crash = obs["ego"]["collision"][0] > 0
    print(
        f"[info] done, steps: {steps}, ego in front: {ego_in_front}, ego crash: {ego_crash}, "
        f"sim time: {info['lap_times'][0]:.2f}, real time: {tf - t0:.2f}"
    )


if __name__ == "__main__":
    import gymnasium as gym
    import gym_envs

    env = gym.make("f110-multi-agent-v0", track_name="Spielberg")
    pp = DummyPlanner(params={"vgain": 4.0})
    run_planner(env, pp, render=True)
