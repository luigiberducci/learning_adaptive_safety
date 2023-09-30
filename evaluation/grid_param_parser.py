import argparse
import pathlib
import shutil
import time
from abc import abstractmethod
from typing import Dict, Union, Tuple, List

import numpy as np
import pandas as pd
import yaml

from rl.crpo import Agent
from training_env_factory import (
    load_env_param_factory,
    load_cbf_param_factory,
    make_env_factory,
)


class ParamGridParser:
    @abstractmethod
    def define_grid_space(self, args) -> np.ndarray:
        pass

    @abstractmethod
    def parse(self, instance: np.ndarray) -> Dict[str, float]:
        pass


class GenericParamGridParser(ParamGridParser):
    def __init__(self):
        self.checkpoints = []
        self.checkpoints_names = []

    def define_grid_space(self, args) -> Tuple[List[str], np.ndarray]:
        gamma_grid = np.linspace(
            args.grid_min_gamma, args.grid_max_gamma, args.grid_gamma_n
        )

        self.checkpoints_names = (
            args.checkpoints_names
            if args.checkpoints_names
            else [f"agent{i}" for i in range(len(args.checkpoints))]
        )
        self.checkpoints = [
            load_checkpoint(
                env_id=args.env_id,
                checkpoint=checkpoint,
                checkpoint_name=name,
                render_mode=args.render_mode,
                outdir=args.outdir,
            )[0]
            for checkpoint, name in zip(args.checkpoints, self.checkpoints_names)
        ]

        params_names, params_grids = [], []
        for grid_param in args.grid_params:
            assert isinstance(grid_param, str)
            tokens = grid_param.split("=")
            assert (
                len(tokens) == 2
            ), "expected format: param_name=value1,value2,...,valueN"
            param_name, param_values_str = tokens[0], tokens[1]

            param_values = []
            for v in param_values_str.split(","):
                try:
                    v = float(v)
                except ValueError:
                    v = str(v)
                assert isinstance(v, (float, str)), "expected float or string"
                param_values.append(v)

            assert len(param_values) > 0, "expected at least one value"
            params_names.append(param_name)
            params_grids.append(param_values)

        self.params_names = params_names

        if len(self.checkpoints) > 0:
            gamma_grid = self.checkpoints
        else:
            gamma_grid = list(gamma_grid)

        n_grids = len(params_grids)
        grid = np.array(np.meshgrid(gamma_grid, *params_grids)).T.reshape(
            -1, n_grids + 1
        )

        return ["gamma"] + self.params_names, grid

    def parse(self, instance: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Return a string identifier and a dictionary of parameters.

        :param instance: 1d array of parameters
        :return: (id string, dict of parameters)
        """
        # instance: [gamma, other params]
        gamma, others = instance[0], instance[1:]

        if isinstance(gamma, float):
            gamma_name = f"{gamma:.2f}"
        else:
            cp_idx = self.checkpoints.index(gamma)
            gamma_name = self.checkpoints_names[cp_idx]

        param_str = f"gamma{gamma_name}"
        # concatenate other params as name, value
        for name, value in zip(self.params_names, others):
            if isinstance(value, float):
                param_str += f"_{name}{value:.2f}"
            else:
                param_str += f"_{name}{value}"

        return param_str, {
            "gamma": gamma,
            "gamma_name": gamma_name,
            **{name: value for name, value in zip(self.params_names, others)},
        }


# helper functions
def load_checkpoint(
    env_id: str,
    checkpoint: Union[str, pathlib.Path],
    checkpoint_name: str = None,
    render_mode: str = None,
    outdir: Union[str, pathlib.Path] = None,
) -> Agent:
    if isinstance(checkpoint, str):
        checkpoint = pathlib.Path(checkpoint)

    assert checkpoint.parent.name in ["checkpoints", "torch_save"] or (
        checkpoint.is_dir() and checkpoint.name in ["checkpoints", "torch_save"]
    ), "assume folder structure is logdir/checkpoints/checkpoint.pt or logdir/checkpoints/"

    # find base dir
    if checkpoint.is_dir():
        cp_dir = checkpoint
        base_dir = cp_dir.parent
    else:
        cp_dir = checkpoint.parent
        base_dir = cp_dir.parent

    if cp_dir.name == "checkpoints":
        checkpoint_type = "cleanrl"
        eval_filename = "evaluations.csv"
        gby_key = "global_step"
        checkpoints_regex = "model_*_*steps.pt"
        config_filename = "args.yaml"
        checkpoint_to_key = lambda x: int(x.stem.split("_")[-1].replace("steps", ""))
        reward_field = "episodic_returns"
        cost_field = "episodic_costs"
    elif cp_dir.name == "torch_save":
        checkpoint_type = "omnisafe"
        eval_filename = "progress.csv"
        gby_key = "Train/Epoch"
        checkpoints_regex = "epoch-*.pt"
        config_filename = "config.json"
        checkpoint_to_key = lambda x: float(x.stem.split("-")[-1])
        reward_field = "Metrics/EpRet"
        cost_field = "Metrics/EpCost"
    else:
        raise NotImplementedError(
            f"checkpoint folder name {checkpoint.name} not supported"
        )

    # load model
    cp_filepath = None
    if checkpoint.is_dir():
        # load checkpoint.parent/evaluations.csv with return, cost, lenght, global_step
        # for each cp model_*_global_step.pt, average return, cost, length over evaluations with global_step
        # finally, select the one with highest return and lowest cost as cp_filepath
        eval_file = checkpoint.parent / eval_filename
        eval_data = pd.read_csv(eval_file, header=0)

        # gby global_step and average over returns, costs, lengths
        eval_data = eval_data.groupby(gby_key).mean()

        # sweep over checkpoints and select the one with highest return and lowest cost
        best_return, best_cost = -np.inf, np.inf
        # sort checkpoints by global_step
        all_files = sorted(
            checkpoint.glob(checkpoints_regex),
            key=checkpoint_to_key,
        )
        for cp in all_files:
            cp_key = checkpoint_to_key(cp)

            try:
                cp_return = eval_data.loc[cp_key][reward_field]
                cp_cost = eval_data.loc[cp_key][cost_field]
            except KeyError:
                continue

            if cp_return >= best_return and cp_cost <= best_cost:
                best_return, best_cost = cp_return, cp_cost
                cp_filepath = cp.absolute()
    else:
        cp_filepath = checkpoint

    assert cp_filepath is not None, f"could not find checkpoint from {checkpoint}"

    # copy checkpoint to out_dir if specified
    if outdir is not None:
        cp_name = (
            checkpoint_name
            if checkpoint_name is not None
            else f"checkpoint{int(time.time())}"
        )
        outdir = pathlib.Path(outdir / "checkpoints" / cp_name)
        outdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cp_filepath, outdir)

        # args.yaml
        shutil.copy(base_dir / config_filename, outdir)

    if checkpoint_type == "cleanrl":
        if (base_dir / config_filename).exists():
            train_args = argparse.Namespace(
                **yaml.load(
                    (base_dir / config_filename).read_text(), Loader=yaml.SafeLoader
                )
            )
        else:
            raise FileNotFoundError(f"Could not find args.yaml in {base_dir}")

        env_params = load_env_param_factory(env_id=env_id)(train_args)
        cbf_params = load_cbf_param_factory(env_id=env_id)(train_args)

        envs = make_env_factory(env_id=env_id)(
            env_id=train_args.env_id,
            env_params=env_params,
            cbf_params=cbf_params,
            idx=0,
            capture_video=False,
            log_dir=None,
            evaluation=True,
            default_render_mode=render_mode,
        )()

        agent = Agent(envs, train_args.rpo_alpha)
        agent.load(envs, cp_filepath, map_location="cpu")
        agent_fn = lambda obs, deterministic: agent.plan(
            obs, deterministic=deterministic
        )

    elif checkpoint_type == "omnisafe":
        import omnisafe
        from gym_envs import omnisafe_adapter

        evaluator = omnisafe.Evaluator()
        log_dir = str(cp_filepath.parent.parent)
        render_mode = "rgb_array" if render_mode is None else render_mode
        evaluator.load_saved(
            save_dir=log_dir,
            model_name=cp_filepath.name,
            render_mode=render_mode,
        )

        envs, actor = evaluator._env, evaluator._actor
        agent_fn = lambda obs, deterministic: actor.predict(
            obs, deterministic=deterministic
        )
        print(f"[debug] loading omnisafe checkpoint: {cp_filepath.name}")

    return agent_fn, envs


def grid_param_parser_factory(env_id: str) -> ParamGridParser:
    return GenericParamGridParser()
