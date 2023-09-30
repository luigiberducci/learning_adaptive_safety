import numpy as np

from gym_envs.multi_agent_env.rewards.reward_fn import RewardFn


class WinLoseRewardFn(RewardFn):
    """
    Implements a sparse reward function that rewards the agent for winning and penalizes it for losing.
    The agent with respect to which the reward is computed is specified by the 'agent_id' parameter.

    Additionally, the reward function can be configured to penalize the agent for crashing and to
    penalize the agent for taking high-magnitude steering actions.
    """

    def __init__(
        self,
        env,
        win_reward: float = 1.0,
        lose_reward: float = -0.5,
        crash_penalty: float = 0.0,
        action_reg: float = 0.0,
        agent_id: str = "ego",
        **kwargs
    ):
        super().__init__(env, **kwargs)
        self.env = env
        self.agent_id = agent_id

        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.crash_penalty = crash_penalty
        self.action_reg = action_reg

    def __call__(self, state, action, next_state, done):
        assert (
            self.agent_id in next_state
        ), "state must contain observation for agent_id"
        assert (
            "collision" in next_state[self.agent_id]
        ), "state must contain collision observation"
        assert (
            "frenet_coords" in next_state[self.agent_id]
        ), "state must contain frenet_coords observation"
        assert "steering" in action, "action must contain steering"

        reward = 0.0

        if next_state[self.agent_id]["collision"] > 0:
            reward += self.crash_penalty

        if done and next_state[self.agent_id]["collision"] <= 0:
            agent_s = next_state[self.agent_id]["frenet_coords"][0]
            opp_ids = [
                opp_id for opp_id in next_state.keys() if opp_id != self.agent_id
            ]

            n_npcs = len(self.env.agents_ids) - 1
            ego_rank = n_npcs - np.count_nonzero(
                [agent_s > next_state[opp_id]["frenet_coords"][0] for opp_id in opp_ids]
            )

            norm_rank = ego_rank / n_npcs
            reward += self.win_reward * (1 - norm_rank) + self.lose_reward * (norm_rank)

        reward += self.action_reg * np.linalg.norm(action["steering"])

        return reward
