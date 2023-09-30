import math
import pathlib
from typing import List, Tuple

import gymnasium
import numpy as np
import pygame
from gymnasium.spaces import Box
from pydantic.utils import deep_update

from gym_envs.particle_env.common.car import Car
from gym_envs.particle_env.common.control import get_trajectory, filter_output_primal
from gym_envs.particle_env.common.gp_controller import CBFCar
from gym_envs.particle_env.observations import observation_factory

data_dir = pathlib.Path(__file__).parent.absolute() / "data"


class ParticleEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, params={}):
        super().__init__()

        # env params
        self.params = {
            "T": 150,  # simulation time
            "world_size": 30.0,  # size of the world
            "cell_size": 5.0,  # size of each cell in the grid map
            "min_agents": 1,
            "max_agents": 10,
            "dist_threshold": 1.0,
            "min_dist_0": 8.0,
            "min_dist_goal": 15.0,
            "coll_threshold": 5.0,
            "horizon_set": [0, 1],
            "goal_lookahead": 1.0,
            "noise_a": 0.001,  # noise on acceleration input
            "init_type": "random",  # random, fixed
            "reward_type": "norm_progress",  # negative_dist, norm_progress, success
            "obs_type": {"type": "raw"},  # grid, raw
            "cbf_gamma_range": [0.0, 1.0],  # range of cbf coefficient gamma
            "interactive_agents": True,  # if False, agents do not move
            "ctrl_params": {
                "use_clf": True,
                "use_cbf": True,
                "robust": True,
                "opt_decay": False,
                "odcbf_gamma_penalty": 1e5,
                "safety_dim": 1,  # number of cbf coefficients for increasing distance from ego
            },
        }
        self.params = deep_update(self.params, params)
        self.world_size = self.params["world_size"]
        self.cell_size = self.params["cell_size"]

        # interval variables
        self.min_dist = None
        self.N_a = None
        self.agents = None
        self.agents_ctrl = None
        self.ego_controller = None
        self.reward_type = self.params["reward_type"]

        # obs and action spaces
        self.obs_type = observation_factory(env=self, **self.params["obs_type"])
        self.observation_space = self.obs_type.space()

        action_dim = int(
            0 if self.params["ctrl_params"]["use_clf"] else 2
        )  # 2 means goal-conditioned (x, y)
        action_dim += int(
            float(self.params["ctrl_params"]["use_cbf"])
            * self.params["ctrl_params"]["safety_dim"]
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(action_dim,))

        # rendering
        self.window_size = 1024
        self.ticks = 60
        self.render_mode = render_mode
        self.clock = None
        self.screen = None

        assert (
            self.render_mode in self.metadata["render_modes"]
            or self.render_mode is None
        )

    def reset(self, seed=None, options=None):
        # seeding
        if seed is not None:
            np.random.seed(seed)
        super().reset(seed=seed)

        # initialize agents
        min_agents = self.params["min_agents"]
        max_agents = self.params["max_agents"]

        if options is not None:
            if "n_agents" in options:
                min_agents = max_agents = options["n_agents"]

            # override params
            for k in options:
                if k in self.params:
                    self.params[k] = options[k]

        self.N_a = np.random.randint(min_agents, max_agents + 1)

        assert self.N_a is not None, "number of agents not initialized"
        assert (
            self.params["min_agents"] <= self.N_a <= self.params["max_agents"]
        ), "invalid number of agents"

        # initialize starting and goal positions
        positions = []
        for i in range(self.N_a):
            other_starting_positions = np.array(positions)
            x0, y0 = self._sample_free_position(
                position_to_avoid=other_starting_positions,
                dist_avoid=self.params["min_dist_0"],
                i=i,
            )
            positions.append([x0, y0])
        positions = np.array(positions)

        goals = []
        for i in range(self.N_a):
            positions_to_avoid = np.array(goals)
            xf, yf = self._sample_free_position(
                position_to_avoid=positions_to_avoid,
                position_to_be_far=positions[i][None],
                dist_avoid=self.params["min_dist_0"],
                dist_far=self.params["min_dist_goal"],
                i=i,
            )
            goals.append([xf, yf])
        goals = np.array(goals)

        # reset interval variables
        self.min_dist = np.inf
        self.steps = 0

        # initialize starting and goal positions
        self.agents = []
        self.agents_ctrl = np.zeros((self.N_a, 2))

        for i in range(self.N_a):
            x0, y0 = positions[i]
            goal = goals[i]

            if i == 0:
                agent = CBFCar(x0, y0, N_a=self.N_a, params=self.params["ctrl_params"])
                agent.goal = goal
                agent.max_acceleration = 8.0
            else:
                agent = Car(x0, y0)
                agent.goal = goal

            self.agents.append(agent)

        # Set barrier for each agent
        horizon_set = self.params["horizon_set"]
        if isinstance(horizon_set, float):
            horizon_set = [horizon_set]

        for i in range(1, self.N_a):
            if self.params["init_type"] == "random":
                idx = np.random.randint(len(horizon_set))
            else:
                idx = i % len(horizon_set)
            self.agents[i].Ds = horizon_set[idx]

        # store starting state
        self.start_state = self.agents[0].state.copy()

        obs = self.obs_type.observe()
        info = {}

        # debug
        # print(f"[reset] state: {self.agents[0].state}, goal: {self.agents[0].goal}")

        return obs, info

    def _sample_free_position(
        self,
        position_to_avoid: np.ndarray,
        position_to_be_far: np.ndarray = None,
        dist_avoid: float = 1.0,
        dist_far: float = 1.0,
        max_iters: int = 100,
        i: int = 0,
    ) -> np.ndarray:
        """
        Samples a position in the environment at random, and ensures
        that the position is at least a distance dist away from any
        occupied position.

        :param position_to_avoid: A list of positions that are already occupied, shape (K, 2).
        :param position_to_be_far: A list of positions that the sampled position should be far from, shape (K, 2).
        :param dist_avoid: The minimum distance to keep from occupied positions.
        :param dist_far: The minimum distance to keep from positions to be far from.
        :param max_iters: The maximum number of iterations to try.
        :param i: The index of the agent to be placed.
        :return: an array of shape (2,) representing the sampled position.
        """

        if self.params["init_type"] == "random" or i == 0:
            max_x = max_y = (
                self.world_size - self.cell_size
            )  # sample within the world with a small margin to the border
            in_collision = True
            x, y = -1, -1
            while in_collision and max_iters > 0:
                in_collision = False
                max_iters -= 1

                # sample position along the border
                if np.random.rand() < 0.5:
                    x = np.random.choice([0, max_x])
                    y = max_y * np.random.rand()
                else:
                    x = max_x * np.random.rand()
                    y = np.random.choice([0, max_y])

                if position_to_avoid.shape[0] > 0:
                    dists = np.linalg.norm(position_to_avoid - np.array([x, y]), axis=1)
                    if np.any(dists < dist_avoid):
                        in_collision = True

                if position_to_be_far is not None and position_to_be_far.shape[0] > 0:
                    dist = np.linalg.norm(position_to_be_far - np.array([x, y]))
                    if dist < dist_far:
                        in_collision = True
        elif self.params["init_type"] == "fixed":
            if position_to_be_far is None:
                # starting pose
                sign = 1 if i % 2 == 0 else -1
                x = float(i / (self.N_a - 1)) * self.world_size
                y = ((self.world_size) / 2.0) + sign * ((self.world_size - 2.0) / 2.0)
            else:
                # goal pose
                sign = 1 if i % 2 == 1 else -1
                i = self.N_a - i - 1
                x = float(i / (self.N_a - 1)) * self.world_size
                y = ((self.world_size) / 2.0) + sign * ((self.world_size - 2.0) / 2.0)

        else:
            raise ValueError(f"Unknown init_type: {self.params['init_type']}")

        return np.array([x, y])

    def process_action(self, action):
        # extract goal, if use_clf
        if not self.params["ctrl_params"]["use_clf"]:
            goal = self.agents[0].state[:2] + action[:2] * self.params["goal_lookahead"]
            action = action[2:]
        else:
            goal = None

        # extract gamma, if use_cbf
        if self.params["ctrl_params"]["use_cbf"]:
            min_gamma, max_gamma = self.params["cbf_gamma_range"]
            gammas = (action + 1) / 2.0 * (max_gamma - min_gamma) + min_gamma
        else:
            gammas = None

        return goal, gammas

    def step(self, action):
        current_ego_state = self.agents[
            0
        ].state.copy()  # store current state for potential shaping

        goal, gamma = self.process_action(action)

        rel_state = np.zeros((self.N_a, 4))
        rel_state[0, :] = self.agents[0].state
        d_state = np.zeros((self.N_a, 4))

        # compute input for other agents
        for j in range(1, self.N_a):
            if self.params["interactive_agents"]:
                # Obtain (CBF) controller for other agent (if applicable)
                try:
                    u2, x2_path, x2_0 = get_trajectory(self.agents[j])
                    if self.agents[j].Ds > 0:
                        u2, _ = filter_output_primal(j, self.agents, x2_path)
                except ValueError as e:
                    u2 = np.zeros(2)
                    x2_0 = self.agents[j].state
            else:
                # Zero controller for other agent
                u2 = np.zeros(2)
                x2_0 = self.agents[j].state

            self.agents_ctrl[j] = u2
            # get agent's relative state
            rel_state[j, :] = x2_0 - self.agents[0].state
            # update min dist
            self.min_dist = min(self.min_dist, np.linalg.norm(rel_state[j, :2]))

        # compute input for our robot
        self.agents_ctrl[0], opt_info = self.agents[0].plan(
            state=rel_state,
            d_state=d_state,
            agents=self.agents,
            goal=goal,
            gamma=gamma,
        )

        # step simulation
        noise_a = self.params["noise_a"]
        for j in range(self.N_a):
            xj = self.agents[j].state

            noisy_ctrl = self.agents_ctrl[j] + noise_a * (np.random.rand(2) - 0.5)
            self.agents[j].update(noisy_ctrl)

            p, v = self.agents[j].fh_err(xj)
            d_state[j, :] = self.agents[j].state - np.concatenate((p, v))

        # check termination conditions
        coll_threshold = self.params["coll_threshold"]
        dist_threshold = self.params["dist_threshold"]
        success, collision_flag = False, False
        for j in range(1, self.N_a):
            if (
                np.linalg.norm(self.agents[0].position - self.agents[j].position)
                < coll_threshold
            ):
                success = False
                collision_flag = True
        if (
            np.linalg.norm(self.agents[0].position - self.agents[0].goal)
            < dist_threshold
            and collision_flag
        ):
            success = False
        elif (
            np.linalg.norm(self.agents[0].position - self.agents[0].goal)
            < dist_threshold
            and not collision_flag
        ):
            success = True

        # reward, info, obs
        cost = float(collision_flag)
        self.steps += 1
        obs = self.obs_type.observe()
        info = {
            "min_dist": self.min_dist,
            "unfeasible": int(opt_info["opt_status"] < 0),
            "success": int(success),
            "collision": int(collision_flag),
            "cost": cost,
            "cost_sum": cost,
            "gamma": gamma,
        }
        reward = self.reward(
            state=current_ego_state, next_state=self.agents[0].state, info=info
        )
        done = self.steps >= self.params["T"] or success or collision_flag
        truncated = collision_flag

        return obs, reward, done, truncated, info

    def reward(self, state, next_state, info):
        if self.reward_type == "negative_dist":
            Q = 0.001 * np.diag(
                [10.0, 10.0, 1.0, 1.0]
            )  # add scaling 0.001 to keep step-reward in +-10
            goal = np.array([self.agents[0].goal[0], self.agents[0].goal[1], 0.0, 0.0])
            reward = -(next_state - goal).T @ Q @ (next_state - goal)
        elif self.reward_type == "success":
            reward = float(info["success"])
        elif self.reward_type == "norm_progress":
            dist_0 = np.linalg.norm(self.start_state[:2] - self.agents[0].goal)
            dist_state = np.linalg.norm(state[:2] - self.agents[0].goal)
            dist_next_state = np.linalg.norm(next_state[:2] - self.agents[0].goal)
            reward = (dist_state - dist_next_state) / (dist_0 + 1e-6)
        return reward

    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
                self.clock = pygame.time.Clock()

        ppu = self.window_size / self.world_size

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((220, 220, 220))
        # Draw other agents
        radius = self.params["coll_threshold"] / 2 * ppu
        for j in range(self.N_a):
            if j == 0:
                color = [0, 0, 200]
            else:
                color = [200, 0, 0]

            pygame.draw.circle(
                canvas,
                color,
                (self.agents[j].position * ppu).astype(int),
                radius,
            )
            # add label in the center with agent index
            font = pygame.font.Font(None, 100)
            text = font.render(f"{j}", 1, (10, 10, 10))
            textpos = text.get_rect()
            textpos.centerx = self.agents[j].position[0] * ppu
            textpos.centery = self.agents[j].position[1] * ppu
            canvas.blit(text, textpos)

        # Draw our goal
        agent1_goal = pygame.image.load(f"{data_dir}/star.png")
        rect = agent1_goal.get_rect()

        for j in range(self.N_a):
            # if you want to see other agents' goals, remove the if statement
            if j > 0:
                break

            agent = self.agents[j]
            # draw goal
            canvas.blit(
                agent1_goal,
                agent.goal * ppu - (rect.width / 2, rect.height / 2),
            )
            # add label in the center with agent index
            font = pygame.font.Font(None, 100)
            text = font.render(f"{j}", 1, (10, 10, 10))
            textpos = text.get_rect()
            textpos.centerx = agent.goal[0] * ppu
            textpos.centery = agent.goal[1] * ppu
            canvas.blit(text, textpos)

        if self.render_mode == "human":
            self.screen.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        pass

    @property
    def state(self):
        states = [agent.state for agent in self.agents]
        goals = [agent.goal for agent in self.agents]
        joint_state = np.concatenate((states, goals), axis=1)
        return joint_state


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    n_episodes = 10
    n_agents = 1
    seed = 321
    render_mode = "human"
    save = False
    plot_obs = False
    gamma = 0.0

    env = ParticleEnv(
        render_mode=render_mode,
        params={
            "world_size": 30.0,
            "init_type": "random",
            "ctrl_params": {
                "use_clf": False,
                "use_cbf": True,
                "robust": False,
                "opt_decay": False,
            },
        },
    )
    print(env.observation_space.shape)

    # simulation
    if plot_obs:
        fig, axes = plt.subplots(1, 4, figsize=(10, 5))

    mean_cost = 0.0
    mean_reward = 0.0

    t0 = time.time()
    for i in range(n_episodes):
        done = False
        obs, _ = env.reset(seed=seed + i, options={"n_agents": n_agents})

        ep_length = 0
        ep_unfeasible_k = 0
        tot_reward, tot_cost = 0.0, 0.0

        while not done:
            # action as normalized gamma, from 0-1 to -1 to 1
            action = env.action_space.sample()
            action[-1] = 2 * gamma - 1.0

            obs, reward, done, truncated, info = env.step(action)
            tot_reward += reward
            tot_cost += info["cost"]
            frame = env.render()

            if save and render_mode == "rgb_array" and ep_length % 10 == 0:
                plt.imsave(f"frame_{i}_{ep_length}.png", frame)

            ep_length += 1
            ep_unfeasible_k += info["unfeasible"]

            # plot observation
            if not plot_obs:
                continue

            assert (
                env.params["obs_type"] == "grid"
            ), "plotting is only supported for grid observations"
            for i, (ax, feature) in enumerate(
                zip(
                    axes.flatten(),
                    ["occupancy", "velx", "vely", "ego_goal", None],
                )
            ):
                if feature is None:
                    ax.axis("off")
                    continue
                ax.cla()
                ax.imshow(obs[:, :, i], cmap="gray")
                ax.set_title(feature)
            plt.pause(0.01)

        if save and render_mode == "rgb_array":
            plt.imsave(f"frame_{i}_{ep_length}.png", frame)

        # printout
        min_dist = info["min_dist"]
        mean_cost += tot_cost
        mean_reward += tot_reward
        print(
            f"\tepisode {i}: success: {info['success']}, collision: {info['collision']}, "
            f"length: {ep_length}, reward: {tot_reward:.2f}, cost: {tot_cost:.2f}"
        )

    print(
        f"[result] mean cost: {mean_cost / n_episodes:.2f}, mean reward: {mean_reward / n_episodes:.2f}"
    )
    print(f"[done] time: {time.time() - t0:.2f}")
    env.close()
