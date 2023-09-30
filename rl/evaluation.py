from typing import Dict, Any, Union

import gymnasium as gym
import torch


def evaluate(
    agent,
    env: gym.Env,
    num_episodes: int = 10,
    seed: int = None,
    verbose: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    assert isinstance(env.unwrapped, gym.vector.VectorEnv) and env.num_envs == 1

    episode_returns = []
    episode_sum_costs = []
    episode_lengths = []

    for i in range(num_episodes):
        obs, infos = env.reset(seed=seed + i if seed is not None else None)
        done = False
        episode_rewards = []
        episode_costs = []
        while not done:
            with torch.no_grad():
                tobs = torch.from_numpy(obs).float().to(device)  # shape (1, obs_dim)
                tactions, _, _, _ = agent.get_action_and_value(tobs, deterministic=True)
                actions = tactions.cpu().numpy()

            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            # env.call("render")

            if "final_info" in infos:
                cost = [infos["final_info"][0]["cost_sum"]]
            else:
                cost = infos["cost_sum"]

            episode_rewards.append(float(rewards[0]))
            episode_costs.append(float(cost[0]))

            obs = next_obs
            done = terminated or truncated

        episode_returns.append(sum(episode_rewards))
        episode_sum_costs.append(sum(episode_costs))
        episode_lengths.append(len(episode_rewards))

        if verbose:
            print(
                f"\tepisode {i}: return: {episode_returns[-1]:.2f}, "
                f"cost: {episode_sum_costs[-1]:.2f}, length: {episode_lengths[-1]}"
            )

    assert (
        len(episode_returns) == num_episodes
    ), f"len(episode_returns) {len(episode_returns)} != num_episodes {num_episodes}"
    assert (
        len(episode_sum_costs) == num_episodes
    ), f"len(episode_costs) {len(episode_sum_costs)} != num_episodes {num_episodes}"
    assert (
        len(episode_lengths) == num_episodes
    ), f"len(episode_lengths) {len(episode_lengths)} != num_episodes {num_episodes}"

    return {
        "episodic_returns": episode_returns,
        "episodic_costs": episode_sum_costs,
        "episodic_lengths": episode_lengths,
    }
