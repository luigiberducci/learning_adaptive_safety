import argparse
import csv
import os
import pathlib
import random
import time
import zipfile
from collections import deque
from distutils.util import strtobool
from typing import Union, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from gym_envs.multi_agent_env.planners.planner import Planner
from rl.evaluation import evaluate


def make_crpo_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="the logging directory"
    )

    parser.add_argument("--seed", type=int, default=None, help="seed of the experiment")
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="save checkpoint of the agent every # update",
    )
    parser.add_argument(
        "--evaluation-freq",
        type=int,
        default=10,
        help="evaluate the agent every # update",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=20,
        help="the number of episodes for evaluating the agent",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--env-id",
        type=str,
        default="particle-env-v0",
        help="the id of the environment",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches", type=int, default=32, help="the number of mini-batches"
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=10,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--norm-cost",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles costs normalization",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )

    # rpo loss
    parser.add_argument(
        "--ent-coef", type=float, default=0.0, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="coefficient of the value function"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )
    parser.add_argument(
        "--rpo-alpha", type=float, default=0.1, help="the alpha parameter for RPO"
    )

    # constrained optimization
    parser.add_argument(
        "--lagrange-mult",
        type=float,
        default=1.0,
        help="lagrangian multiplier for the performance constraint",
    )
    parser.add_argument(
        "--lagrange-mult-max",
        type=float,
        default=100.0,
        help="max value for lagrange multiplier",
    )

    # pid lagrangian
    parser.add_argument(
        "--use-pid-lagrangian",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use pid lagrangian to tune the penalty parameter.",
    )
    parser.add_argument(
        "--pid-lagrangian-gains",
        type=float,
        nargs=3,
        default=[1.0, 0.1, 0.1],
        help="pid gains for tuning lagrangian multiplier",
    )
    parser.add_argument(
        "--pid-lagrangian-ema-alpha",
        type=float,
        default=0.95,
        help="alpha coefficient for exponential moving average of P-D terms",
    )
    parser.add_argument(
        "--pid-lagrangian-d-delay",
        type=int,
        default=1,
        help="delay for derivative term in pid lagrangian",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=0.05,
        help="cost limit for constrained policy optimization",
    )

    parser.add_argument(
        "--verbose",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles whether or not to print training updates",
    )

    return parser


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module, Planner):
    def __init__(self, envs, rpo_alpha):
        super().__init__()
        self.rpo_alpha = rpo_alpha

        # check vectorized gym_envs
        if getattr(envs, "is_vector_env", False):
            observation_space = envs.single_observation_space
            action_space = envs.single_action_space
        else:
            observation_space = envs.observation_space
            action_space = envs.action_space

        # build policy and value networks
        use_cnn = len(observation_space.shape) == 3
        if use_cnn:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, int(np.prod(action_space.shape))), std=0.01),
            )
            self.critic = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 1), std=1),
            )
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.array(observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, int(np.prod(action_space.shape))), std=0.01),
            )

        self.actor_logstd = nn.Parameter(
            torch.zeros(1, int(np.prod(action_space.shape)))
        )

    def get_value(self, x):
        if len(x.shape) > 3:
            # image observation, channel first
            x = x.permute(0, 3, 1, 2)

        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic: bool = False):
        if len(x.shape) > 3:
            # image observation, channel first
            x = x.permute(0, 3, 1, 2)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None and deterministic:
            action = action_mean
        elif action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = (
                torch.FloatTensor(action_mean.shape)
                .uniform_(-self.rpo_alpha, self.rpo_alpha)
                .to(action_mean.device)
            )
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )

    def save(self, envs, save_path: Union[pathlib.Path, str]):
        with zipfile.ZipFile(save_path, mode="w") as archive:
            with archive.open(
                "model_weights.pth", mode="w", force_zip64=True
            ) as pytorch_variables_file:
                torch.save(self.state_dict(), pytorch_variables_file)

            if hasattr(envs, "obs_rms"):
                with archive.open(
                    "obs_normalizer.pth", mode="w", force_zip64=True
                ) as param_file:
                    obs_rms = envs.obs_rms
                    data = {
                        "mean": obs_rms.mean,
                        "var": obs_rms.var,
                        "count": obs_rms.count,
                    }
                    torch.save(data, param_file)

    def load(self, envs, load_path: Union[pathlib.Path, str], map_location=None):
        with zipfile.ZipFile(load_path) as archive:
            with archive.open("model_weights.pth", mode="r") as weight_file:
                weights = torch.load(weight_file, map_location=map_location)
                self.load_state_dict(weights)

            if hasattr(envs, "obs_rms"):
                with archive.open("obs_normalizer.pth", mode="r") as param_file:
                    data = torch.load(param_file)
                    for v in ["mean", "var", "count"]:
                        setattr(envs.obs_rms, v, data[v])

    def plan(self, obs: np.ndarray, deterministic: bool = True) -> Dict[str, float]:
        with torch.no_grad():
            tobs = torch.from_numpy(obs[None]).float()  # shape (1, obs_dim)
            taction, _, _, _ = self.get_action_and_value(
                tobs, deterministic=deterministic
            )
            action = taction.numpy()[0]
        return action

    def reset(self, **kwargs):
        pass


def run_crpo(args, make_env_fn: callable, env_params: dict = {}, cbf_params: dict = {}):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    if args.seed is None:
        args.seed = np.random.randint(0, 1000000)

    start_time = time.time()

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    log_dir = pathlib.Path(args.log_dir) / run_name

    # logger
    writer = SummaryWriter(f"{log_dir}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # write args to file
    with open(f"{log_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    cp_dir = pathlib.Path(f"{log_dir}/checkpoints")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env_fn(
                env_id=args.env_id,
                env_params=env_params,
                cbf_params=cbf_params,
                idx=i,
                evaluation=False,
                capture_video=False,
                log_dir=log_dir,
            )
            for i in range(args.num_envs)
        ]
    )
    # reward normalization only for training env
    envs = gym.wrappers.NormalizeReward(envs, gamma=0.99)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    if len(envs.observation_space.shape) <= 2:  # flatten observation
        envs = gym.wrappers.NormalizeObservation(envs)
        envs = gym.wrappers.TransformObservation(
            envs, lambda obs: np.clip(obs, -10, 10)
        )

    eval_envs = gym.vector.AsyncVectorEnv(
        [
            make_env_fn(
                env_id=args.env_id,
                env_params=env_params,
                cbf_params=cbf_params,
                idx=0,
                evaluation=True,
                capture_video=args.capture_video,
                log_dir=log_dir,
            )
        ]
    )
    eval_envs = gym.wrappers.NormalizeObservation(eval_envs)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))

    # sanity check
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    print("[info] observation space:", envs.single_observation_space)
    print("[info] action space:", envs.single_action_space)

    agent = Agent(envs, args.rpo_alpha).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # PID Lagrangian
    p_int = 0.0  # integral term
    dt_queue = deque(maxlen=args.pid_lagrangian_d_delay)  # queue for derivative term
    dt_queue.append(0.0)
    ema_p = 0.0  # exponential moving average for proportional term
    ema_d = 0.0  # exponential moving average for derivative term
    alpha_ema_p = (
        alpha_ema_d
    ) = args.pid_lagrangian_ema_alpha  # ema alpha for proportional and derivative terms
    lagrange_mult = args.lagrange_mult  # lagrange multiplier for performance constraint

    # start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    ep_sum_costs = np.zeros((args.num_envs))
    ep_steps = np.zeros((args.num_envs))

    for update in range(1, num_updates + 1):
        sum_batch_costs, n_completed_episodes = 0, 0
        ep_t0 = [time.time() for _ in range(args.num_envs)]

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(done).to(device),
            )

            # new part - compute safety rewards
            if "final_info" not in infos:
                step_costs = torch.tensor(infos["cost_sum"]).to(device).view(-1)
            else:
                step_costs = []
                for i, info in enumerate(infos["final_info"]):
                    if info is None:
                        step_costs.append(infos["cost_sum"][i])
                    else:
                        step_costs.append(info["cost_sum"])
                step_costs = torch.tensor(step_costs).to(device).view(-1)
            costs[step] = step_costs

            ep_sum_costs += step_costs.cpu().numpy()
            ep_steps += 1

            # envs.call('render')

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for i, info in enumerate(infos["final_info"]):
                # Skip the gym_envs that are not done
                if info is None:
                    continue
                ep_return, ep_length = info["episode"]["r"], info["episode"]["l"]
                ep_cost = ep_sum_costs[i]

                # log data
                writer.add_scalar("train/episodic_return", ep_return, global_step)
                writer.add_scalar("train/episodic_cost", ep_cost, global_step)
                writer.add_scalar("train/episodic_length", ep_length, global_step)

                if "success" in info["episode"]:
                    writer.add_scalar(
                        "train/episodic_success", float(info["success"]), global_step
                    )

                if "cbf_stats" in info:
                    for k, v in info["cbf_stats"].items():
                        writer.add_scalar(f"stats_cbf/{k}", v, global_step)

                # print logs
                if args.verbose:
                    print(
                        f"global_step={global_step}, episodic_length={ep_length}, "
                        f"episodic_return={ep_return[0]:.2f}, episodic_cost={ep_cost:.2f}, "
                        f"lagrange_mult={lagrange_mult:.2f}, ep time={time.time() - ep_t0[i]:.2f} sec"
                    )

                # update batch statistics for update lagrange multiplier
                n_completed_episodes += 1
                sum_batch_costs += ep_cost

                # reset safety, steps
                ep_sum_costs[i] = 0
                ep_steps[i] = 0
                ep_t0[i] = time.time()

        # advantage estimation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)

            rtgs = torch.zeros_like(rewards).to(device)
            cost_rtgs = torch.zeros_like(costs).to(device)

            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                    nextrtg = torch.zeros_like(next_value).to(device)
                    nextsafetyrtg = torch.zeros_like(next_value).to(device)
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                # reward to go estimation
                rtgs[t] = rewards[t] + args.gamma * nextrtg * nextnonterminal
                nextrtg = rtgs[t]

                # cost to go estimation
                cost_rtgs[t] = costs[t] + args.gamma * nextsafetyrtg * nextnonterminal
                nextsafetyrtg = cost_rtgs[t]

                # gae estimation
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )

            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_cost_rtgs = cost_rtgs.reshape(-1)

        # lagrangian multiplier
        if args.use_pid_lagrangian:
            # adaptive penalty coefficient - pid lagrangian
            kp, ki, kd = args.pid_lagrangian_gains
            cost_limit = args.cost_limit  # cost limit: sum_t c_t < cost_limit

            # compute the error
            avg_ep_cost = sum_batch_costs / n_completed_episodes
            p_err = avg_ep_cost - cost_limit

            p_int = max(0.0, p_int + p_err * ki)

            ema_p = alpha_ema_p * ema_p + (1 - alpha_ema_p) * p_err
            ema_d = alpha_ema_d * ema_d + (1 - alpha_ema_d) * (avg_ep_cost)

            p_der = max(0.0, ema_d - dt_queue[0])

            # compute the penalty coefficient
            lagrange_mult = kp * ema_p + ki * p_int + kd * p_der
            lagrange_mult = max(0.0, min(lagrange_mult, args.lagrange_mult_max))
        else:
            # use constant multiplier - standard regularization
            lagrange_mult = args.lagrange_mult

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # bugfix: when mb_inds is of length 1, advantage normalization produces nan
                # then in that case, skip
                if len(mb_inds) == 1:
                    continue

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                mb_cadvantages = b_cost_rtgs[mb_inds]
                if args.norm_cost:
                    mb_cadvantages = (mb_cadvantages - mb_cadvantages.mean()) / (
                        mb_cadvantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Cost loss
                cost_loss1 = mb_cadvantages * ratio
                cost_loss2 = mb_cadvantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                cost_loss = torch.min(cost_loss1, cost_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    + lagrange_mult * cost_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                pass

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "train/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("train/lagrange_mult", lagrange_mult, global_step)

        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/cost_loss", cost_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        writer.add_scalar(
            "train/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        print("SPS:", int(global_step / (time.time() - start_time)))

        if update % args.checkpoint_freq == 0:
            if not cp_dir.exists():
                cp_dir.mkdir(parents=True, exist_ok=True)
            agent.save(envs, f"{cp_dir}/model_{update}_{global_step}steps.pt")

        if update == 1 or update % args.evaluation_freq == 0:
            # if using obs normalizer, ensure normalization statistics are the same in eval envs
            if hasattr(envs, "obs_rms"):
                for k in ["mean", "var", "count"]:
                    val = getattr(envs.obs_rms, k)
                    setattr(eval_envs.obs_rms, k, val)

            # evaluate agent performance
            print(
                f"\n[info] evaluation: global steps: {global_step}, num eval episodes: {args.num_eval_episodes}, seed: {args.seed}"
            )

            with torch.no_grad():
                eval_stats = evaluate(
                    agent,
                    eval_envs,
                    args.num_eval_episodes,
                    seed=args.seed,
                    device=device,
                    verbose=True,
                )

            # log aggregated evaluation statistics
            print(f"[info] evaluation: global steps: {global_step}, ", end="")
            for k, v in eval_stats.items():
                aggregated_mu, aggregated_std = np.mean(v), np.std(v)
                writer.add_scalar(f"eval/{k}", aggregated_mu, global_step)
                print(f"{k}: {aggregated_mu:.2f} +- {aggregated_std:.2f}, ", end="")
            print()

            # concatenate individual evaluation statistics to json file evaluations.csv
            eval_stats["global_step"] = [
                global_step for _ in eval_stats["episodic_returns"]
            ]
            if not (log_dir / "evaluations.csv").exists():
                header = ",".join(list(eval_stats.keys()))
                with open(log_dir / "evaluations.csv", "w") as f:
                    f.write(header + "\n")

            with open(log_dir / "evaluations.csv", "a") as f:
                write = csv.writer(f)
                for i in range(len(eval_stats["episodic_returns"])):
                    row = [eval_stats[k][i] for k in eval_stats.keys()]
                    write.writerow(row)

    if not cp_dir.exists():
        cp_dir.mkdir(parents=True, exist_ok=True)
    agent.save(envs, f"{cp_dir}/model_final.pt")

    envs.close()
    eval_envs.close()
    writer.close()

    print(f"[done] elapsed time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    # by default, use double_integrator environment
    from run_training_only_adaptive import make_particle_env
    import gym_envs

    # base parser
    parser = make_crpo_parser()

    # env specific arguments
    parser.add_argument(
        "--reward-type", type=str, default="norm_progress", help="the type of reward"
    )
    parser.add_argument(
        "--obs-type", type=str, default="state", help="the type of reward"
    )
    parser.add_argument(
        "--time-limit", type=float, default=10.0, help="the time limit of each episode"
    )

    parser.add_argument("--cbf-type", type=str, default="advanced")
    parser.add_argument(
        "--safety-dim",
        type=int,
        default=1,
        help="number of coefficients in adaptive cbf",
    )
    args = parser.parse_args()

    env_params = {
        "time_limit": args.time_limit,
        "reward_type": args.reward_type,
        "observation_type": args.obs_type,
    }
    cbf_params = {
        "use_cbf": True,
        "cbf_type": "advanced",
        "safety_dim": 1,
        "cbf_gamma_range": [0.1, 1.0],
    }

    # run training
    run_crpo(
        args=args,
        env_params=env_params,
        cbf_params=cbf_params,
        make_env_fn=make_particle_env,
    )
