from typing import Tuple

import numpy as np

from gym_envs.multi_agent_env.rewards.reward_fn import RewardFn


class ProgressRewardFn(RewardFn):
    def __init__(
        self,
        progress_range: Tuple[float, float] = (-1000, 1000),
        normalize: bool = False,
    ):
        self.min_progress, self.max_progress = progress_range
        self.normalize = normalize

    def __call__(self, state, action, next_state, done):
        assert "ego" in next_state, "next_state must contain ego observation"
        assert (
            "progress" in next_state["ego"]
        ), "next_state must contain progress observation"

        progress = next_state["ego"]["progress"][0]

        if self.normalize:
            progress = np.clip(progress, self.min_progress, self.max_progress)
            progress = (progress - self.min_progress) / (
                self.max_progress - self.min_progress
            )

        return progress


class TwoPlayersProgressRewardFn(RewardFn):
    def __call__(self, state, action, next_state, done):
        assert (
            "ego" in state and "ego" in next_state
        ), "state/next_state must contain ego observation"
        assert (
            "npc0" in state and "npc0" in next_state
        ), "state/next_state must contain npc0 observation"

        distance = state["ego"]["frenet_coords"][0] - state["npc0"]["frenet_coords"][0]
        next_distance = (
            next_state["ego"]["frenet_coords"][0]
            - next_state["npc0"]["frenet_coords"][0]
        )

        progress = next_distance - distance
        return progress


class TwoPlayersDistanceRewardFn(RewardFn):
    def __call__(self, state, action, next_state, done):
        assert (
            "ego" in state and "ego" in next_state
        ), "state/next_state must contain ego observation"
        assert (
            "npc0" in state and "npc0" in next_state
        ), "state/next_state must contain npc0 observation"

        rel_distance = (
            next_state["ego"]["frenet_coords"][0]
            - next_state["npc0"]["frenet_coords"][0]
        )

        return rel_distance


class MultiPlayersSparseRelDistanceRewardFn(RewardFn):
    def __call__(self, state, action, next_state, done):
        opp_ids = [k for k in state.keys() if k != "ego"]
        assert (
            "ego" in state and "ego" in next_state
        ), "state/next_state must contain ego observation"
        assert len(opp_ids) > 0, "state/next_state must contain npc information"

        tot_rel_distance = 0
        for opp_id in opp_ids:
            rel_distance = (
                next_state["ego"]["frenet_coords"][0]
                - next_state[opp_id]["frenet_coords"][0]
            )
            rel_distance = np.tanh(
                rel_distance / 5.0
            )  # to normalize the reward within reasonable range, +-1 per agent
            tot_rel_distance += rel_distance

        reward = tot_rel_distance if done else 0.0
        return reward


class WinLooseProgressShapingRewardFn(RewardFn):
    def __init__(
        self,
        env,
        win_reward: float = 1.0,
        lose_reward: float = -0.5,
        crash_penalty: float = 0.0,
        action_reg: float = 0.0,
        min_progress: float = -1000,
        max_progress: float = 1000,
        agent_id: str = "ego",
    ):
        self.min_progress, self.max_progress = min_progress, max_progress

        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.crash_penalty = crash_penalty
        self.action_reg = action_reg

        self.agent_id = agent_id
        self.opponent_id = None

    def __call__(self, state, action, next_state, done):
        if self.opponent_id is None:
            opp_ids = [
                opp_id for opp_id in next_state.keys() if opp_id != self.agent_id
            ]
            assert len(opp_ids) == 1, "only one opponent is supported"
            self.opponent_id = opp_ids[0]

        assert all(
            [agent_id in state for agent_id in [self.agent_id, self.opponent_id]]
        ), "agent obs not found in state"
        assert all(
            [agent_id in next_state for agent_id in [self.agent_id, self.opponent_id]]
        ), "agent obs not found in next_state"

        rel_progress = (
            state[self.agent_id]["frenet_coords"][0]
            - state[self.opponent_id]["frenet_coords"][0]
        )
        rel_progress_next = (
            next_state[self.agent_id]["frenet_coords"][0]
            - next_state[self.opponent_id]["frenet_coords"][0]
        )

        # base reward = 1 if win, 0 if lose
        base_reward = 1.0 if done and rel_progress_next > 0 else 0.0

        # potential-based shaping
        if not done:
            shaping = rel_progress_next - rel_progress
        else:
            shaping = -rel_progress

        return base_reward + shaping
