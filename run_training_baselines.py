from distutils.util import strtobool

import numpy as np
import omnisafe
import gym_envs.omnisafe_adapter
from training_env_factory import get_env_id


def main(args):
    algo = args.algo
    cost_limit = args.cost_limit
    env_id = args.env_id
    use_cbf = args.use_cbf
    use_ctrl = args.use_ctrl
    seed = args.seed
    total_timesteps = args.total_timesteps
    num_envs = args.num_envs
    logdir = args.log_dir
    checkpoint_freq = args.checkpoint_freq
    entropy_coef = args.entropy_coef

    custom_cfgs = {
        "seed": seed,
        "train_cfgs": {
            "total_steps": total_timesteps,
            "vector_env_nums": num_envs,
            "parallel": 1,
        },
        "algo_cfgs": {
            "use_cost": True,
            "steps_per_epoch": 2048,
        },
        "logger_cfgs": {
            "use_wandb": False,
            "use_tensorboard": True,
            "save_model_freq": checkpoint_freq,
            "log_dir": logdir,
            "window_lens": 100,
        },
    }

    # entropy coef
    if algo in ["PPOLag", "PPOPID"]:
        custom_cfgs["algo_cfgs"]["entropy_coef"] = entropy_coef
        custom_cfgs["algo_cfgs"]["update_iters"] = 10
        custom_cfgs["algo_cfgs"]["use_max_grad_norm"] = True
        custom_cfgs["algo_cfgs"]["max_grad_norm"] = 0.5

    # algo-specific configs
    if algo.endswith("Lag"):
        custom_cfgs["lagrange_cfgs"] = {
            "cost_limit": cost_limit,
            "lagrangian_multiplier_init": 1.0,
            "lambda_lr": 5e-2,
        }
    elif algo.endswith("PID"):
        custom_cfgs["lagrange_cfgs"] = {
            "cost_limit": cost_limit,
            "pid_kp": 1.0,
            "pid_ki": 0.01,
            "pid_kd": 0.01,
        }
    elif algo == "PPOSaute":
        custom_cfgs["algo_cfgs"]["safety_budget"] = cost_limit
    else:
        custom_cfgs["algo_cfgs"]["cost_limit"] = cost_limit

    custom_env_id = get_env_id(env_id, use_cbf, use_ctrl)
    agent = omnisafe.Agent(algo, custom_env_id, custom_cfgs=custom_cfgs)
    agent.learn()


if __name__ == "__main__":
    import argparse

    envs = [
        "particle-env-v0",
        "particle-env-v1",
        "f110-multi-agent-v0",
        "f110-multi-agent-v1",
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo", type=str, default="PPOPID", help="Baseline algorithm to use"
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=0.1,
        help="Cost limit for Lagrange multiplier",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="Entropy coefficient for exploration bonus",
    )

    parser.add_argument("--env-id", type=str, default="particle-env-v0", choices=envs)
    parser.add_argument(
        "--use-ctrl",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles the use of a controller",
    )
    parser.add_argument(
        "--use-cbf",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles the use of a CBF",
    )
    parser.add_argument(
        "--use-decay",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=False,
        help="Toggles the use of optimal-decay coefficient in CBF",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for the environment"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1024000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of vectorized environments"
    )

    parser.add_argument(
        "--log-dir", type=str, default="logs/baselines", help="Directory to save logs"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Save agent model every N updates",
    )

    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 1000000)

    main(args)
