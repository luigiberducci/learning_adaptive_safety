from functools import partial

import gymnasium as gym

# env registration
gym.register(
    "f110-multi-agent-v0", entry_point="gym_envs.multi_agent_env:MultiAgentRaceEnv"
)
gym.register("particle-env-v0", entry_point="gym_envs.particle_env:ParticleEnv")


# cbf registration
def cbf_factory(env_id: str, cbf_type: str):
    if env_id == "f110-multi-agent-v0":
        from gym_envs.multi_agent_env.multi_agent_env_cbfs import cbf_factory_f1tenth

        return partial(cbf_factory_f1tenth, cbf_type=cbf_type)
    raise ValueError(f"cbf for env_id {env_id} is not supported")
