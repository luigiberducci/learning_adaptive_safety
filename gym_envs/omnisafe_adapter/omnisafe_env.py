import pathlib
from typing import Any

import safety_gymnasium
import torch
import yaml
from gymnasium.spaces import Box
from omnisafe.envs import env_register, SafetyGymnasiumEnv

from training_env_factory import make_env_factory


def make_safe_env(
    env_id: str, env_params: dict, cbf_params: dict, idx: int, render_mode: str
):
    def thunk():
        env = make_env_factory(env_id)(
            env_id=env_id,
            env_params=env_params,
            cbf_params=cbf_params,
            idx=idx,
            evaluation=False,
            capture_video=False,
            log_dir=False,
            default_render_mode=render_mode,
        )()
        return safety_gymnasium.wrappers.Gymnasium2SafetyGymnasium(env)

    return thunk


@env_register
class MyEnv(SafetyGymnasiumEnv):
    _support_envs = [
        "particle-env-auto-simple-v0",
        "particle-env-auto-simpledecay-v0",
        "particle-env-rl-none-v0",
        "particle-env-rl-simple-v0",
        "particle-env-auto-simple-v1",
        "particle-env-auto-simpledecay-v1",
        "particle-env-rl-none-v1",
        "particle-env-rl-simple-v1",
        "f110-multi-agent-auto-cbf-v0",
        "f110-multi-agent-auto-cbfdecay-v0",
        "f110-multi-agent-rl-none-v0",
        "f110-multi-agent-rl-cbf-v0",
        "f110-multi-agent-auto-cbf-v1",
        "f110-multi-agent-auto-cbfdecay-v1",
        "f110-multi-agent-rl-none-v1",
        "f110-multi-agent-rl-cbf-v1",
    ]

    def __init__(
        self, env_id: str, num_envs: int = 1, device: str = "cpu", **kwargs: Any
    ) -> None:
        self._num_envs = num_envs
        self._device = torch.device(device)
        self._options = None

        # open yaml file in ../cfgs/env_id.yaml and load env_params, cbf_params
        cfg_path = pathlib.Path(__file__).parent.parent / "cfgs" / f"{env_id}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Env config file {cfg_path} does not exist.")

        with open(cfg_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            base_env_id = data["base_env"]
            env_params = data["env_params"]
            cbf_params = data["cbf_params"]

        render_mode = kwargs.get("render_mode", None)
        if num_envs > 1:
            self._env = safety_gymnasium.vector.SafetyAsyncVectorEnv(
                [
                    make_safe_env(
                        env_id=base_env_id,
                        env_params=env_params,
                        cbf_params=cbf_params,
                        idx=i,
                        render_mode=render_mode,
                    )
                    for i in range(self._num_envs)
                ]
            )

            assert isinstance(
                self._env.single_action_space, Box
            ), "Only support Box action space."
            assert isinstance(
                self._env.single_observation_space,
                Box,
            ), "Only support Box observation space."
            self._action_space = self._env.single_action_space
            self._observation_space = self._env.single_observation_space
        else:
            self.need_time_limit_wrapper = True
            self.need_auto_reset_wrapper = True
            self._env = make_safe_env(
                env_id=base_env_id,
                env_params=env_params,
                cbf_params=cbf_params,
                idx=0,
                render_mode=render_mode,
            )()
            assert isinstance(
                self._env.action_space, Box
            ), "Only support Box action space."
            assert isinstance(
                self._env.observation_space,
                Box,
            ), "Only support Box observation space."
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        self._metadata = self._env.metadata

    def set_options(self, options: dict) -> None:
        self._options = options

    def reset(self, seed=None):
        if self._options:
            obs, info = self._env.reset(seed=seed, options=self._options)
            return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info
        else:
            return super().reset(seed=seed)

    def __getattr__(self, item):
        return getattr(self._env, item)


if __name__ == "__main__":
    import numpy as np
    import omnisafe

    env_id = "particle-env-auto-simple-v0"
    seed = np.random.randint(0, 1000)

    custom_cfgs = {
        "seed": seed,
        "train_cfgs": {
            "total_steps": 1000,
            "vector_env_nums": 1,
            "parallel": 1,
        },
        "algo_cfgs": {
            "use_cost": True,
            "steps_per_epoch": 256,
        },
        "lagrange_cfgs": {
            "cost_limit": 0.1,
        },
        "logger_cfgs": {
            "use_wandb": False,
            "use_tensorboard": True,
            "save_model_freq": 3,
            "log_dir": "./runs",
            "window_lens": 100,
        },
    }
    agent = omnisafe.Agent("PPOLag", env_id, custom_cfgs=custom_cfgs)
    agent.learn()

    agent.plot(smooth=1)
    agent.render(num_episodes=1, render_mode="rgb_array", width=256, height=256)
