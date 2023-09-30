from typing import Tuple, Union, List

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, ActType


class CBFSafetyLayer(gym.Wrapper):
    """
    Extend action space with adaptive cbf coefficients, ie. gammas.

    On step, assume action to be composed by raw action and safety gammas.
    Then, separate action from gammas, and use CBF to project nominal action onto safe control set.
    """

    def __init__(
        self,
        env,
        safety_dim: int,
        gamma_range: Union[Tuple[float, float], List, np.ndarray],
        make_cbf: callable,
        alpha: float = 1.0,
    ):
        """

        :param env: the environment to wrap
        :param safety_dim: the number of safety coefficients to control (ie., gammas in cbf constraint)
        :param alpha: the class-K function, scalar for simplicity
        :param gamma_range: the range of safety coefficients, [min_gamma, max_gamma]
        :param make_cbf: the function to make cbf projection
        """
        super().__init__(env=env)
        self.safety_dim = safety_dim
        self.min_gamma, self.max_gamma = gamma_range
        self.alpha = alpha
        self.use_old_api = isinstance(env, gym.Env)
        self.cbf_project = make_cbf(env=self)

        low_action = self.action_space.low
        high_action = self.action_space.high
        shape_action = self.action_space.shape

        self.action_space = gym.spaces.Box(
            low=np.concatenate(
                (low_action, -np.ones(self.safety_dim, dtype=np.float32))
            ),
            high=np.concatenate(
                (high_action, np.ones(self.safety_dim, dtype=np.float32))
            ),
            shape=(shape_action[0] + self.safety_dim,),
        )

        # for statistics
        self.hxs = {}
        self.slacks = {}
        self.opt_status = []
        self.corrections = []

    def reset(self, **kwargs):
        self.hxs = {}
        self.slacks = {}
        self.opt_status = []
        self.corrections = []
        return self.env.reset(**kwargs)

    def step(self, action: WrapperActType) -> ActType:
        assert (
            action.shape == self.action_space.shape
        ), "action shape must match action space shape"

        # separate action from gammas
        nominal_action, norm_gammas = (
            action[: -self.safety_dim],
            action[-self.safety_dim :],
        )

        # denormalize gamma from [-1, +1] to [min_gamma, max_gamma]
        norm_gammas = np.clip(norm_gammas, -1.0, 1.0)
        gammas = (norm_gammas + 1.0) / 2.0 * (
            self.max_gamma - self.min_gamma
        ) + self.min_gamma

        # project nominal action onto safe control set
        action, opt_infos = self.cbf_project(self.state, nominal_action, gammas)

        # step env
        next_state, reward, done, truncated, info = self.env.step(action)

        # extend info
        info.update(
            {
                "action": nominal_action,
                "safe_action": action,
                "gammas": gammas,
            }
        )
        info.update(opt_infos)

        # update logging stats
        for k, v in opt_infos.items():
            if k.startswith("hx"):
                if k not in self.hxs:
                    self.hxs[k] = []
                self.hxs[k].append(v)
            elif k.startswith("slack"):
                if k not in self.slacks:
                    self.slacks[k] = []
                self.slacks[k].append(v)
        self.corrections.append(np.linalg.norm(nominal_action - action))
        self.opt_status.append(info["opt_status"])

        if done:
            assert "cbf_stats" not in info, "cbf_stats already in info"
            info["cbf_stats"] = {}

            for k, v in self.hxs.items():
                min_hx, max_hx, mean_hx = np.min(v), np.max(v), np.mean(v)
                info["cbf_stats"][f"episodic_{k}_min"] = min_hx
                info["cbf_stats"][f"episodic_{k}_max"] = max_hx
                info["cbf_stats"][f"episodic_{k}_mean"] = mean_hx

            for k, v in self.slacks.items():
                min_slack, max_slack, mean_slack = np.min(v), np.max(v), np.mean(v)
                info["cbf_stats"][f"episodic_{k}_min"] = min_slack
                info["cbf_stats"][f"episodic_{k}_max"] = max_slack
                info["cbf_stats"][f"episodic_{k}_mean"] = mean_slack

            min_c, max_c, mean_c = (
                np.min(self.corrections),
                np.max(self.corrections),
                np.mean(self.corrections),
            )
            info["cbf_stats"][f"episodic_correction_min"] = min_c
            info["cbf_stats"][f"episodic_correction_max"] = max_c
            info["cbf_stats"][f"episodic_correction_mean"] = mean_c

            min_s, max_s, mean_s = (
                np.min(self.opt_status),
                np.max(self.opt_status),
                np.mean(self.opt_status),
            )
            info["cbf_stats"][f"episodic_opt_status_min"] = min_s
            info["cbf_stats"][f"episodic_opt_status_max"] = max_s
            info["cbf_stats"][f"episodic_opt_status_mean"] = mean_s

        return next_state, reward, done, truncated, info


def test_cbf_wrapper_double_integrator():
    from gym_envs.double_integrator import DoubleIntegratorEnv
    from gym_envs import cbf_factory

    env_id = "double-integrator-v0"
    safety_dim = 1
    env = DoubleIntegratorEnv()
    gamma_range = [0.0, 1.0]
    make_cbf = cbf_factory(env_id=env_id, cbf_type="advanced")

    env = CBFSafetyLayer(
        env,
        safety_dim=safety_dim,
        alpha=1.0,
        gamma_range=gamma_range,
        make_cbf=make_cbf,
    )

    obs, _ = env.reset()
    done = False

    while not done:
        action_gamma = env.action_space.sample()
        action_gamma[:2] = 1.0

        # step the environment
        obs, reward, done, truncated, info = env.step(action_gamma)
        env.render()

    env.close()

    print(info["cbf_stats"]["episodic_hx_min"])
    assert info["cbf_stats"]["episodic_hx_min"] > -1e-3, "hx_min should be positive"


def test_cbf_wrapper_f110():
    from gym_envs import cbf_factory
    from gym_envs.multi_agent_env import MultiAgentRaceEnv
    from gym_envs.wrappers import FlattenAction

    env_id = "f110-multi-agent-v0"
    env = MultiAgentRaceEnv(
        track_name="General1",
        params={"termination": {"types": ["on_timeout"], "timeout": 10.0}},
    )
    env = FlattenAction(env)
    safety_dim = 2
    make_cbf = cbf_factory(env_id=env_id, cbf_type="advanced")
    gamma_range = [0.0, 1.0]

    env = CBFSafetyLayer(
        env,
        safety_dim=safety_dim,
        alpha=1.0,
        gamma_range=gamma_range,
        make_cbf=make_cbf,
    )

    obs, _ = env.reset()
    done = False

    while not done:
        action_gamma = env.action_space.sample()
        action_gamma[:2] = 1.0

        # step the environment
        obs, reward, done, truncated, info = env.step(action_gamma)
        # env.render()

    env.close()

    print("min hx_left", info["cbf_stats"]["episodic_hx_left_min"])
    print("min hx_right", info["cbf_stats"]["episodic_hx_right_min"])

    assert (
        info["cbf_stats"]["episodic_hx_left_min"] > -1e-3
    ), "hx_left_min should be positive"
    assert (
        info["cbf_stats"]["episodic_hx_right_min"] > -1e-3
    ), "hx_right_min should be positive"


if __name__ == "__main__":
    from gym_envs.double_integrator import DoubleIntegratorEnv

    test_cbf_wrapper_double_integrator()
