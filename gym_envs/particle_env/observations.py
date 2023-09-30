import math
from abc import abstractmethod

import numpy as np
from gymnasium.spaces import Box


class Observation:
    """
    Abstract class for observations. Each observation must implement the space and observe methods.

    :param env: The environment.
    """

    def __init__(self, env: "ParticleEnv"):
        self.env = env

    @property
    @abstractmethod
    def space(self):
        raise NotImplementedError()

    @abstractmethod
    def observe(self):
        raise NotImplementedError()


class GridObservation(Observation):
    def space(self):
        width = height = math.ceil(self.env.world_size / self.env.cell_size)
        return Box(low=0, high=255, shape=(width, height, 4), dtype=np.uint8)

    def observe(self):
        width = height = math.ceil(self.env.world_size / self.env.cell_size)
        occupancy_grid = np.zeros((width, height))
        velocity_x_grid = np.zeros((width, height))
        velocity_y_grid = np.zeros((width, height))
        is_ego_grid = np.zeros((width, height))
        ego_goal_grid = np.zeros((width, height))

        agents = self.env.agents
        x0, y0 = agents[0].position

        for i in range(len(agents)):
            x, y = agents[i].position

            # relative to ego, s.t. ego is in the center at width/2, height/2
            x = math.floor((x - x0) / self.env.cell_size + width / 2)
            y = math.floor((y - y0) / self.env.cell_size + height / 2)

            if not (0 <= x < width and 0 <= y < height):
                continue

            occupancy_grid[x, y] = 1.0
            velocity_x_grid[x, y] = (
                0.1 + 0.9 * agents[i].velocity[0] / agents[i].max_velocity
            )
            velocity_y_grid[x, y] = (
                0.1 + 0.9 * agents[i].velocity[1] / agents[i].max_velocity
            )
            if i == 0:
                is_ego_grid[x, y] = 1.0
                goal_x, goal_y = agents[i].goal

                # relative to ego, s.t. ego is in the center at width/2, height/2
                goal_x = math.floor((goal_x - x0) / self.env.cell_size + width / 2)
                goal_y = math.floor((goal_y - y0) / self.env.cell_size + height / 2)

                if not (0 <= goal_x < width and 0 <= goal_y < height):
                    continue

                ego_goal_grid[goal_x, goal_y] = 1.0

        observation = np.stack(
            [
                occupancy_grid.T,
                velocity_x_grid.T,
                velocity_y_grid.T,
                ego_goal_grid.T,
            ],
            axis=2,
        )
        observation = (255 * observation).astype(
            np.uint8
        )  # scale and convert to uint8 to use cnn policy
        return observation


class FeatureObservation(Observation):
    def __init__(
        self, env, n_obs_agents: int = 7, sorted: bool = True, relative: bool = True
    ):
        super().__init__(env)
        self.n_obs_agents = n_obs_agents
        self.obs_dim = 6  # x, y, vx, vy, gx, gy
        self.relative = relative
        self.sorted = sorted

    def space(self):
        return Box(
            low=-1000,
            high=1000,
            shape=(self.n_obs_agents, self.obs_dim),
            dtype=np.float32,
        )

    def observe(self):
        # compute observation as stack of `n_obs_agents` states,
        # where the first agent is the ego state relative to the goal, and
        # the others states are sorted by distance to the ego and relative to it

        # sort agents by distance to ego
        agents_state_goal = np.array(
            [(a.position, a.velocity, a.goal) for a in self.env.agents]
        )

        if self.sorted:
            agents_state_goal = sorted(
                agents_state_goal,
                key=lambda x: np.linalg.norm(x[0] - agents_state_goal[0][0]),
            )

        # stack states and goals of agents into observation
        obs = np.zeros((self.n_obs_agents, self.obs_dim))
        for i in range(self.n_obs_agents):
            if i > len(agents_state_goal) - 1:
                # no more agents (ie., n agents < n_obs_agents)
                break

            obs[i, :2] = agents_state_goal[i][0]
            obs[i, 2:4] = agents_state_goal[i][1]
            obs[i, 4:] = agents_state_goal[i][2]

        # compute observation positions and goals relative to ego position
        if self.relative:
            ego_position = agents_state_goal[0][0]
            obs[:, :2] -= ego_position
            obs[:, 4:] -= ego_position

        return obs.astype(np.float32)


def observation_factory(env, type: str, **kwargs) -> Observation:
    if type == "grid":
        return GridObservation(env, **kwargs)
    elif type == "raw":
        return FeatureObservation(env, **kwargs)
    else:
        raise ValueError(f"Unknown observation type: {type}")
