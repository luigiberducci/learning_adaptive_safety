from gym_envs.multi_agent_env.rewards.reward_fn import RewardFn


def reward_fn_factory(env, reward_type: str, **kwargs) -> RewardFn:
    if reward_type == "default":
        from gym_envs.multi_agent_env.rewards.reward_fn import DefaultRewardFn

        return DefaultRewardFn(env)
    if reward_type == "progress":
        from gym_envs.multi_agent_env.rewards.progress_reward import ProgressRewardFn

        return ProgressRewardFn(**kwargs)
    if reward_type == "norm_progress":
        from gym_envs.multi_agent_env.rewards.progress_reward import ProgressRewardFn

        return ProgressRewardFn(normalize=True, **kwargs)
    if reward_type == "rel_progress":
        from gym_envs.multi_agent_env.rewards.progress_reward import (
            TwoPlayersProgressRewardFn,
        )

        return TwoPlayersProgressRewardFn(env, **kwargs)
    if reward_type == "rel_distance":
        from gym_envs.multi_agent_env.rewards.progress_reward import (
            TwoPlayersDistanceRewardFn,
        )

        return TwoPlayersDistanceRewardFn(env, **kwargs)
    if reward_type == "tracking":
        from gym_envs.multi_agent_env.rewards.tracking_reward import TrackingRewardFn

        return TrackingRewardFn(env, **kwargs)
    elif reward_type == "two_agents_advantage":
        from gym_envs.multi_agent_env.rewards.tracking_reward import (
            TwoAgentsAdvantageRewardFn,
        )

        return TwoAgentsAdvantageRewardFn(**kwargs)
    elif reward_type == "win_lose":
        from gym_envs.multi_agent_env.rewards.sparse_win_lose_reward import (
            WinLoseRewardFn,
        )

        return WinLoseRewardFn(env, agent_id="ego", **kwargs)
    elif reward_type == "win_lose_shaped":
        from gym_envs.multi_agent_env.rewards.progress_reward import (
            WinLooseProgressShapingRewardFn,
        )

        return WinLooseProgressShapingRewardFn(env, agent_id="ego", **kwargs)
    elif reward_type == "sparse_rel_progress":
        from gym_envs.multi_agent_env.rewards.progress_reward import (
            MultiPlayersSparseRelDistanceRewardFn,
        )

        return MultiPlayersSparseRelDistanceRewardFn(env, **kwargs)
    else:
        raise ValueError("unknown reward type: {}".format(reward_type))
