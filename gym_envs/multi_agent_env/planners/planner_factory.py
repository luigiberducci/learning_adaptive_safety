import pathlib
import re
import warnings
from typing import Callable, List

import numpy as np
import yaml

from gym_envs.multi_agent_env.common.track import Track
from gym_envs.multi_agent_env.planners.follow_the_gap import FollowTheGap
from gym_envs.multi_agent_env.planners.planner import (
    Planner,
    DummyPlanner,
    RandomPlanner,
)
from gym_envs.multi_agent_env.planners.pure_pursuit import (
    PurePursuitPlanner,
    AdvancedPurePursuitPlanner,
)
from gym_envs.multi_agent_env.planners.lattice_planner import LatticePlanner
from gym_envs.multi_agent_env.planners.frenet_planner import FrenetPlannerWeightedCosts

PLANNERS_DICT = {
    "random": lambda track, params, agent_id: RandomPlanner(agent_id=agent_id),
    "dummy": lambda track, params, agent_id: DummyPlanner(
        params=params, agent_id=agent_id
    ),
    "ftg": lambda track, params, agent_id: FollowTheGap(
        params=params, agent_id=agent_id
    ),
    "pp": lambda track, params, agent_id: PurePursuitPlanner(
        track=track, params=params, agent_id=agent_id
    ),
    "app": lambda track, params, agent_id: AdvancedPurePursuitPlanner(
        track=track, params=params, agent_id=agent_id
    ),
    "lattice": lambda track, params, agent_id: LatticePlanner(
        track=track, params=params, agent_id=agent_id
    ),
    "frenet": lambda track, params, agent_id: FrenetPlannerWeightedCosts(
        track=track, params=params, agent_id=agent_id
    ),
}


def planner_factory(
    planner: str, track: Track, agent_id: str, params: dict = None
) -> Planner:
    """
    Create a planner object based on the planner name, eg. "pp" for pure pursuit, "lattice" for lattice planner, etc.

    We support passing the velocity gain through the planner string, e.g. "lattice-0.9", "pp-0.8"
    Exceptionally, for the rule-lattice planner, we support passing the safety/target tradeoff additionally,
    e.g. "rule_lattice-0.9-0.5". However, all the other parameters must be given in the params dictionary.

    If no parameters are given, we try to load the default parameters from the f1tenth_racetracks_overrides folder,
    or use the default parameters defined in the planner class, if no overrides are found.

    :param planner: string name of the planner
    :param track: track object
    :param agent_id: string identifier of the agent
    :param params: dictionary of parameters
    :return:
    """
    # first identify the planner
    planner_name = re.sub(r"-\d+.\d+", "", planner)
    planner_vgain = re.findall("\d+\.\d+", planner)

    # load params (given or from file+name)
    if params is None:
        params = load_planner_params(track_name=track.spec.name, planner=planner_name)

    # set explicit string params (vgain, safety/target tradeoff)
    if len(planner_vgain) == 0:
        pass
    elif len(planner_vgain) == 1:
        params["vgain"] = float(planner_vgain[0])
    elif len(planner_vgain) == 2:
        params["vgain"] = float(planner_vgain[0])
        params["safety_target_tradeoff"] = float(planner_vgain[1])
    else:
        raise ValueError(
            "Name-based parametrization is only implemented for vgain and safety/target tradeoff"
        )

    # create planner
    for planner, build_fn in PLANNERS_DICT.items():
        if planner_name.startswith(planner):
            return build_fn(track=track, params=params, agent_id=agent_id)

    raise ValueError(
        f"Unknown planner: {planner_name}, expected one of {PLANNERS_DICT.keys()}"
    )


def load_planner_params(track_name: str, planner: str) -> dict:
    base_dir = (
        pathlib.Path(__file__).parent.parent.parent.parent
        / "f1tenth_racetracks_overrides"
    )

    for track_dir in base_dir.iterdir():
        new_track_name = track_dir.name
        if track_name == new_track_name:
            planner_config_file = track_dir / f"{planner}.yaml"
            if planner_config_file.exists():
                params = yaml.load(
                    planner_config_file.read_text(), Loader=yaml.FullLoader
                )
                return params
    return {}


def load_expert_planner_params(
    track_name: str, planner: str, filter_fn: Callable[[np.ndarray], List[int]] = None
) -> dict:
    assert (
        planner == "lattice"
    ), "Loading expert planner params is only implemented for the lattice planner"

    base_dir = pathlib.Path(__file__).parent / ".." / "f1tenth_racetracks_overrides"

    for track_dir in base_dir.iterdir():
        new_track_name = track_dir.name
        if track_name == new_track_name:
            planner_config_file = track_dir / f"{planner}.yaml"
            expert_weights_file = track_dir / f"{track_name}_GTOP_nearpareto.csv"

            if planner_config_file.exists() and expert_weights_file.exists():
                params = yaml.load(
                    planner_config_file.read_text(), Loader=yaml.FullLoader
                )

                # load expert params
                expert_weights = np.loadtxt(
                    expert_weights_file, delimiter=",", skiprows=1
                )

                if filter_fn is not None:
                    expert_weights = expert_weights[filter_fn(expert_weights)]

                objective_scores = expert_weights[:, :2]
                vgains = expert_weights[:, 2]
                cost_weights = expert_weights[:, 3:]

                # sample id of a random expert
                idx = np.random.randint(len(vgains))
                params["vgain"] = vgains[idx]
                params["weights"] = cost_weights[idx]

                return params

    warnings.warn(
        f"Could not find any expert {planner} for track {track_name}, using filter function {filter_fn}"
    )
    return {}
