from typing import Tuple

import numpy as np

from gym_envs.multi_agent_env.rewards.reward_fn import RewardFn


class TrackingRewardFn(RewardFn):
    def __init__(self, env, **kwargs):
        self.Q = np.diag([1.0, 1.0, 0.1])  # velocity range ~ [0, 10]
        self.R = np.diag(
            [1.0, 0.0]
        )  # the second control is the velocity, makes no sense to penalize it

    def __call__(self, state, action, next_state, done):
        assert "ego" in next_state, "state must contain ego observation"
        assert "pose" in next_state["ego"], "state must contain pose"
        assert (
            "nearest_waypoint_rl" in next_state["ego"]
        ), "state must contain nearest waypoint along raceline"

        state = np.array(
            [
                next_state["ego"]["pose"][0],
                next_state["ego"]["pose"][1],
                next_state["ego"]["velocity"][0],
            ]
        )
        control = np.array([action["steering"], action["velocity"]])
        target_state = next_state["ego"]["nearest_waypoint_rl"]

        cost = (state - target_state).T @ self.Q @ (
            state - target_state
        ) + control.T @ self.R @ control
        reward = np.exp(-cost)

        return reward


class TwoAgentsAdvantageRewardFn(TrackingRewardFn):
    def __init__(
        self, weight: float = 1.0, advantage_range: Tuple[float, float] = (0, 5)
    ):
        self.weight = weight
        self.min_advantage, self.max_advantage = advantage_range

    def __call__(self, state, action, next_state, done):
        assert "ego" in state, "state must contain ego observation"
        assert "ego" in next_state, "next_state must contain ego observation"
        assert "npc0" in state, "state must contain npc0 observation"
        assert "npc0" in next_state, "next_state must contain npc0 observation"

        tracking_reward = super().__call__(state, action, next_state, done)

        advantage = (
            next_state["ego"]["frenet_coords"][0] - state["npc0"]["frenet_coords"][0]
        )
        advantage = np.clip(advantage, self.min_advantage, self.max_advantage)
        norm_advantage = (advantage - self.min_advantage) / (
            self.max_advantage - self.min_advantage
        )

        return self.weight * norm_advantage  # + tracking_reward
