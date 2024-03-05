import argparse
import pathlib
import re
import time
from distutils.util import strtobool

import pandas as pd
import torch
import yaml

from evaluation.grid_param_parser import grid_param_parser_factory, load_checkpoint
from evaluation.metric_logger import MetricLogger, logger_factory
from run_training_baselines import get_env_id


def main(args):
    # parse args
    n_episodes = args.n_episodes
    seed = args.seed
    render_mode = args.render_mode
    indir = pathlib.Path(args.indir) if args.indir else None
    regex = args.regex
    exp_id = args.exp_id
    env_id = args.env_id
    metrics = args.metrics
    plot_type = args.plot

    # create logger for metrics and plotting
    logger = logger_factory(env_id=env_id, metrics=metrics)

    # create param parser for grid search
    grid_parser = grid_param_parser_factory(env_id=env_id)

    if indir:
        all_indirs = set([file.parent for file in indir.glob(regex)])

        for ind in all_indirs:
            if "skip" in ind.name:
                continue
            args.outdir = pathlib.Path(ind) if args.save else None
            param_names, all_stats = load_stats_from_dir(
                logger=logger, indir=ind, file_regex=regex
            )
            gbys = [p for p in param_names if p not in ["gamma", "gamma_name"]]
            logger.plot(all_stats, outdir=args.outdir, gbys=gbys, plot_type=plot_type)

            print("[info] written plots to: ", args.outdir)
    else:
        # prepare outdir
        subdir = f"{env_id}_{exp_id}_{int(time.time())}"
        args.outdir = pathlib.Path(args.outdir) / subdir if args.outdir else None

        # load checkpoints and env
        if len(args.checkpoints) > 0:
            # use first checkpoint to load env-params and create environment
            checkpoint = args.checkpoints[0]
            _, env = load_checkpoint(
                env_id=env_id, checkpoint=checkpoint, render_mode=render_mode
            )
        else:
            # load params from args and create environment
            from gym_envs import omnisafe_adapter
            from omnisafe.envs.core import make

            custom_env_id = get_env_id(
                env_id, use_cbf=True, use_ctrl=True, use_decay=args.use_decay
            )
            env = make(custom_env_id, render_mode=render_mode)

        param_names, params_grid = grid_parser.define_grid_space(args)
        all_stats = collect_stats_grid_search(
            env=env,
            params_grid=params_grid,
            n_episodes=n_episodes,
            seed=seed,
            logger=logger,
            grid_parser=grid_parser,
            outdir=args.outdir,
            deterministic=args.deterministic,
        )

        gbys = [p for p in param_names if p not in ["gamma", "gamma_name"]]
        logger.plot(all_stats, outdir=args.outdir, gbys=gbys, plot_type=plot_type)

        print("[info] written plots to: ", args.outdir)


def load_stats_from_dir(
    logger: MetricLogger, indir: pathlib.Path, file_regex: str
) -> dict:
    """
    Load stats from a directory containing previous experiments results in csv format.

    :param indir: path to the directory containing the results
    :param file_regex: regex to filter the files
    :return: a dictionary containing the stats
    """
    all_stats = {}

    for f in indir.glob(file_regex):
        # read csv as dict
        results = pd.read_csv(f).to_dict(orient="list")

        # keep only metrics in logger
        results = {k: v for k, v in results.items() if k in logger.metrics}

        # parse run
        run_id = f.stem
        tokens = run_id.split("_")

        # group them to have string, value pairs
        tokens2 = []
        last_token = ""
        for t in tokens:
            last_token = f"{last_token}_{t}" if len(last_token) > 0 else t
            # check if t contains a numeric part
            num = "".join([c for c in t if c.isnumeric()])
            if len(num) != 0:
                tokens2.append(last_token)
                last_token = ""

        tokens = tokens2

        assert len(tokens) > 1, f"Invalid run id: {run_id}"
        gamma_name = tokens[0].replace("gamma", "")
        gamma_name = re.sub(r"agent[0-9]+", "Ada", gamma_name)

        # parse other tokens using str as param name and float as param value
        params = {}
        for t in tokens[1:]:
            k = "".join([c for c in t if c.isalpha()])
            v = float("".join([c for c in t if c.isnumeric() or c == "."]))
            params[k] = v
        params["gamma"] = gamma_name
        params["gamma_name"] = gamma_name

        aggregated_results = logger.aggregate_all_metrics(results)

        print(f"[results] {run_id}")
        for k, v in aggregated_results.items():
            print(f"\t{k}: {v:.2f}")
        print()

        # append results to all_stats
        for k, v in zip(params.keys(), params.values()):
            if k not in all_stats:
                all_stats[k] = []
            all_stats[k].append(v)

        for k, v in aggregated_results.items():
            if k not in all_stats:
                all_stats[k] = []
            all_stats[k].append(v)

    return params.keys(), all_stats


def collect_stats_grid_search(
    env, params_grid, seed, n_episodes, logger, grid_parser, outdir, deterministic
):
    """
    Collect stats for `n_episodes` for each entry in the given `params_grid`.

    :param params_grid: list of param tuples (gamma, n_agents)
    :param seed: int or None
    :param n_episodes: number of episodes to run for each param tuple
    :param outdir: path to the output directory, or None
    :param render_mode: how to render the simulation, or None
    :return: a dictionary containing the stats
    """
    # create output directory
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

        # save config
        with open(outdir / "args.yaml", "w+") as f:
            yaml.dump(vars(args), f)

    # simulation
    all_stats = {}

    for params in params_grid:
        param_str, dict_params = grid_parser.parse(params)
        metrics_list = collect_stats(
            env,
            params=dict_params,
            logger=logger,
            seed=seed,
            n_episodes=n_episodes,
            deterministic=deterministic,
            verbose=True,
        )
        aggregated_results = logger.aggregate_all_metrics(metrics_list)

        print(f"[results]")
        for k, v in aggregated_results.items():
            print(f"\t{k}: {v:.2f}")
        print()

        # save results to file
        if outdir:
            filename = f"{param_str}.csv"
            df = pd.DataFrame(metrics_list)
            df.to_csv(outdir / filename, index=False)

        # append results to all_stats
        for k, v in dict_params.items():
            if k not in all_stats:
                all_stats[k] = []
            all_stats[k].append(v)

        for k, v in aggregated_results.items():
            if k not in all_stats:
                all_stats[k] = []
            all_stats[k].append(v)

    return all_stats


def collect_stats(
    env,
    params: tuple,
    logger: MetricLogger,
    seed: int,
    n_episodes: int,
    deterministic: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Simulate n_episodes with a given `gamma` and `n_agents` and return the results.

    :param env: the simulation environment
    :param params: tuple of parameters according to the task
    :param logger: the logger to record step metrics and aggregate them
    :param seed: initial seed for the random number generator, or None
    :param n_episodes: the number of episodes to simulate
    :param verbose: whether to print intermediate simulations and results
    :return:
    """
    assert seed is None or seed >= 0, "seed must be greater than 0 or None"
    assert n_episodes > 0, "n_episodes must be greater than 0"

    if verbose:
        print(f"[info] seed: {seed}, n_episodes: {n_episodes}, params: {params}")

    # parse params
    assert "gamma" in params, "gamma must be in params"
    gamma_pi = params["gamma"]
    others_params = {k: v for k, v in params.items() if k != "gamma"}

    all_statistics = None  # dict of lists
    for i in range(n_episodes):
        done = False

        seed = seed + i if seed is not None else None
        if hasattr(env, "set_options"):
            env.set_options(options=others_params)
            obs, _ = env.reset(seed=seed)
        else:
            obs, _ = env.reset(seed=seed, options=others_params)

        t0 = time.time()

        ep_statistics = {k: [] for k in logger.metrics}
        while not done:
            # action as normalized gamma, from 0-1 to -1 to 1
            if isinstance(gamma_pi, float):
                # convert gamma from 0-1 to -1 to 1
                action = 2 * gamma_pi - 1.0
                action = torch.tensor(action, dtype=torch.float32)
            else:
                with torch.no_grad():
                    action = gamma_pi(obs, deterministic=deterministic)
            results = env.step(action)

            if len(results) == 5:
                obs, reward, done, truncated, info = results
            else:
                obs, reward, cost, done, truncated, info = results

            env.render()

            # log metrics
            step_stats = logger.get_metrics(env.state, reward, done, truncated, info)
            for k, v in step_stats.items():
                ep_statistics[k].append(v)

        # aggregate step statistics into episode statistics
        ep_statistics = logger.aggregate_ep_metrics(ep_statistics)

        if verbose:
            print(f"\tepisode {i}: ", end="")
            print(", ".join([f"{k}: {v}" for k, v in ep_statistics.items()]), end=" ")
            print(f"elapsed time: {time.time() - t0:.2f} sec")

        if all_statistics is None:
            all_statistics = {k: [v] for k, v in ep_statistics.items()}
        else:
            for k, v in ep_statistics.items():
                all_statistics[k].append(v)

    return all_statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-id", type=str, default="exp")

    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        help="the checkpoint to evaluate",
        default=[],
    )
    parser.add_argument(
        "--checkpoints-names",
        type=str,
        nargs="+",
        help="the name of the checkpoint to evaluate",
        default=[],
    )

    parser.add_argument("--grid-min-gamma", type=float, default=0.01)
    parser.add_argument("--grid-max-gamma", type=float, default=1.0)
    parser.add_argument("--grid-gamma-n", type=int, default=5)

    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument(
        "--deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, use deterministic actions for trained agents",
    )
    parser.add_argument(
        "--use-decay",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=False,
        help="if toggled, use optimal decay for gamma in CBF",
    )

    parser.add_argument(
        "--indir", type=str, default=None, help="Directory from which to load results"
    )
    parser.add_argument(
        "--regex", type=str, default="*.csv", help="Regex to filter results"
    )
    parser.add_argument(
        "--save", action="store_true", help="Whether to save results to file"
    )
    parser.add_argument(
        "--outdir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="the metrics to evaluate",
        default=None,
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=["default", "bar"],
        default="bar",
        help="the plot to show",
    )
    parser.add_argument("--grid-params", type=str, nargs="+", default=[])
    parser.add_argument(
        "env_id",
        type=str,
        help="Environment ID",
        choices=["particle-env-v0", "particle-env-v1", "f110-multi-agent-v0", "f110-multi-agent-v1"],
    )

    args = parser.parse_args()
    args.cbf_gamma_range = (0.0, 1.0)

    t0 = time.time()

    main(args)

    print(f"[done] elapsed time: {time.time() - t0}")
