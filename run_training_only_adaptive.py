import pathlib
from distutils.util import strtobool

import numpy as np
import yaml

from rl import crpo
from rl.crpo import make_crpo_parser
from training_env_factory import make_env_factory
from training_env_factory import get_env_id


if __name__ == "__main__":
    import argparse

    parser = make_crpo_parser()

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

    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 1000000)

    # todo
    env_id = get_env_id(args.env_id, args.use_cbf, args.use_ctrl)
    cfg_path = pathlib.Path(__file__).parent / "gym_envs" / "cfgs" / f"{env_id}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Env config file {cfg_path} does not exist.")

    with open(cfg_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        base_env_id = data["base_env"]
        env_params = data["env_params"]
        cbf_params = data["cbf_params"]

    args.env_id = base_env_id
    make_env = make_env_factory(env_id=args.env_id)

    crpo.run_crpo(
        make_env_fn=make_env,
        env_params=env_params,
        cbf_params=cbf_params,
        args=args,
    )
