from casadi import casadi
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

import gymnasium as gym
import pygame as pygame
from gymnasium import spaces
import numpy as np

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams["text.latex.preamble"]  = r"\usepackage{cmbright}"

debug = False


class DoubleIntegratorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        time_limit=10.0,
        dt=0.2,
        observation_type: str = "state",
        reward_type: str = "norm_progress",
        render_mode="human",
    ):
        # sim parameters
        self.dt = dt
        self.time_limit = time_limit

        self.start_state = np.array([-5.0, -5.0, 0.0, 0.0])
        self.start_xy = self.start_state[:2]

        self.goal_state = np.array([0.0, 0.0, 0.0, 0.0])
        self.goal_xy = self.goal_state[:2]

        self.obst_xy = np.array([-2.0, -2.25])
        self.obst_r = 1.5  # radius of the obstacle

        self.success_threshold = 1.0  # distance to goal to consider episode successful

        # dynamics
        self.A = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        self.B = np.array(
            [
                [0.5 * self.dt**2, 0],
                [0, 0.5 * self.dt**2],
                [self.dt, 0],
                [0, self.dt],
            ]
        )

        # reward
        if reward_type == "negative_dist":
            self.reward = partial(negative_dist_reward_fn, goal_state=self.goal_state)
        elif reward_type == "norm_progress":
            self.reward = partial(
                norm_progress_reward_fn,
                goal_state=self.goal_state,
                init_state=self.start_state,
            )
        elif reward_type == "success":
            self.reward = partial(
                success_reward_fn,
                goal_state=self.goal_state,
                success_threshold=self.success_threshold,
            )
        else:
            raise ValueError(f"Unknown reward type {reward_type}")

        # action space
        self.action_dim = 2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # observation space
        if observation_type == "state":
            self.observation_space = spaces.Box(
                low=-5.0, high=5.0, shape=(8,), dtype=np.float32
            )
        elif observation_type == "state_and_time":
            self.observation_space = spaces.Box(
                low=np.array([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, 0.0]),
                high=np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0]),
                shape=(9,),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown observation type {observation_type}")
        self.observation_type = observation_type

        # render
        self.window = None
        self.clock = None
        self.size = 10
        self.window_size = 512  # The size of the PyGame window
        self.color = None  # used to color the agent, if None use default color
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.zeros(4, dtype=np.float32)
        self.state[:2] = self.start_xy

        self.time = 0.0
        self.total_reward = 0.0
        self.total_cost = 0.0

        observation = self.observe()

        cost = int(
            (self.state[0] - self.obst_xy[0]) ** 2
            + (self.state[1] - self.obst_xy[1]) ** 2
            - self.obst_r**2
            < 0
        )
        done = self.time >= self.time_limit
        success = (
            done
            and np.linalg.norm(self.state - self.goal_state) < self.success_threshold
        )

        info = {
            "success": success,
            "cost": cost,
            "cost_sum": cost,
            "state": self.state,
            "time": self.time,
        }

        return observation, info

    def step(self, action):
        """
        Step the environment by one timestep.

        :param action: (ax, ay, gamma) acceleration in x and y direction and safety level gamma
        :return: new state, reward, done, info
        """
        nstate = self.A @ self.state + self.B @ action
        nstate = np.clip(nstate, -5.0, 5.0)  # clip state to be within the bounds

        self.time += self.dt

        reward = self.reward(self.state, action, nstate)
        cost = int(
            (nstate[0] - self.obst_xy[0]) ** 2
            + (nstate[1] - self.obst_xy[1]) ** 2
            - self.obst_r**2
            < 0
        )

        done = self.time >= self.time_limit
        truncated = False

        success = (
            done
            and np.linalg.norm(self.state - self.goal_state) < self.success_threshold
        )

        info = {
            "success": success,
            "cost": cost,
            "cost_sum": cost,
            "state": self.state,
            "time": self.time,
        }

        self.total_reward += reward
        self.total_cost += cost
        self.state = nstate.astype(np.float32)

        observation = self.observe()

        return observation, reward, done, truncated, info

    def observe(self):
        if self.observation_type == "state":
            # concatenate state and goal state
            observation = np.concatenate([self.state, self.goal_state]).astype(
                np.float32
            )
        elif self.observation_type == "state_and_time":
            remaining_time = (self.time_limit - self.time) / self.time_limit
            observation = np.concatenate(
                [self.state, self.goal_state, [remaining_time]]
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown observation type {self.observation_type}")
        return observation

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        goal_xy = self.goal_xy + self.size / 2
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (goal_xy) * pix_square_size,
            pix_square_size * self.success_threshold,
        )

        # we draw the obstacle
        obst_xy = self.obst_xy + self.size / 2
        pygame.draw.circle(
            canvas,
            (255, 125, 0),
            (obst_xy) * pix_square_size,
            pix_square_size * self.obst_r,
        )

        # Now we draw the agent
        agent_xy = self.state[:2] + self.size / 2
        # interpolate color according to the last_gamma
        color = (255, 0, 0) if self.color is None else self.color
        pygame.draw.circle(
            canvas,
            color,
            [int(p) for p in (agent_xy) * pix_square_size],
            int(pix_square_size / 3),
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


"""
Reward functions
"""


def negative_dist_reward_fn(state, action, next_state, goal_state):
    """
    Standard negative distance reward function with control regularization.

    :param state: state of the environment [x, y, vx, vy]
    :param action: acceleration in x and y direction
    :param next_state: next state of the environment [x, y, vx, vy]
    :param goal_state: goal state of the environment [x, y, vx, vy]
    :return: scalar reward
    """
    goal_cost = 10 * np.linalg.norm((next_state - goal_state)) ** 2
    control_cost = np.linalg.norm(action) ** 2
    return -(goal_cost + control_cost)


def norm_progress_reward_fn(state, action, next_state, goal_state, init_state):
    """
    Reward function that rewards the agent for making progress towards the goal.

    :param state: state of the environment [x, y, vx, vy]
    :param action: acceleration in x and y direction
    :param next_state: next state of the environment [x, y, vx, vy]
    :return: scalar reward
    """
    current_dist = np.linalg.norm((state - goal_state))
    next_dist = np.linalg.norm((next_state - goal_state))
    norm_progress = (current_dist - next_dist) / np.linalg.norm(
        (init_state - goal_state)
    )
    return norm_progress

def success_reward_fn(state, action, next_state, goal_state, success_threshold):
    """
    Sparse reward function that rewards the agent once it reaches the goal.

    :param state:
    :param action:
    :param next_state:
    :param goal_state:
    :param success_threshold:
    :return: scalar reward
    """
    dist = np.linalg.norm((next_state - goal_state))
    if dist < success_threshold:
        return 1.0
    else:
        return -0.01




def double_integrator_advanced_cbf_project(
    env, state, action, gamma, ming=0.0, maxg=1.0
):
    """
    cbf candidate
    h(x) = ||Dp|| - Dv**2 / (2 a_max) - R >= 0

    where:super().action(action)

        Dp = is the distance between the agent and the obstacle, ie. p - p_obst
        Dv = is the velocity between the agent and the obstacle in the collision direction, ie. dp/||dp|| * v
        R = is the radius of the obstacle

    :param state: state of the environment [x, y, vx, vy]
    :param action: acceleration in x and y direction
    :param gamma: scaling factor for the class k function
    :return:
    """
    # optimizer
    opti = casadi.Opti()

    # decision variables
    u_min, u_max = np.array([env.action_space.low[:2], env.action_space.high[:2]])
    u = opti.variable(2)
    opti.subject_to(u_min <= u)
    opti.subject_to(u <= u_max)

    # slacks
    slack = opti.variable()
    opti.subject_to(slack >= 0)

    # state relation
    x, y, vx, vy = state
    dt = env.dt
    a_max = 1.0  # maximum braking in any direction is >= 1.0

    new_x = x + vx * dt + 0.5 * u[0] * dt**2
    new_y = y + vy * dt + 0.5 * u[1] * dt**2
    new_vx = vx + u[0] * dt
    new_vy = vy + u[1] * dt

    # compute hx
    rel_pos = np.array([x - env.obst_xy[0], y - env.obst_xy[1]])
    norm_pos = np.linalg.norm(rel_pos)
    Dp = rel_pos / norm_pos

    rel_vel = np.array([vx, vy])
    Dv = np.dot(rel_vel, Dp)

    # compute hx_next
    rel_pos_next = casadi.vertcat(new_x - env.obst_xy[0], new_y - env.obst_xy[1])
    norm_pos_next = casadi.norm_2(rel_pos_next)
    Dp_next = rel_pos_next / norm_pos_next

    rel_vel_next = casadi.vertcat(new_vx, new_vy)
    Dv_next = casadi.dot(rel_vel_next, Dp_next)

    if Dv > 1e-3:
        # if the agent is moving away from the obstacle
        hx = norm_pos - env.obst_r
        hx_next = norm_pos_next - env.obst_r
    else:
        # if the agent is moving towards the obstacle
        hx = norm_pos - Dv**2 / (2 * a_max) - env.obst_r
        hx_next = norm_pos_next - Dv_next**2 / (2 * a_max) - env.obst_r

    # add cbf constraint
    opti.subject_to(hx_next - hx + slack >= -gamma * hx)

    # objective
    obj = (u[0] - action[0]) ** 2 + (u[1] - action[1]) ** 2 + 1000 * slack
    opti.minimize(obj)

    p_opts = {"print_time": False, "verbose": False}
    s_opts = {"print_level": 0}
    opti.solver("ipopt", p_opts, s_opts)  # "qpoases", p_opts, s_opts)

    try:
        sol = opti.solve()
        safe_input = sol.value(u)

        dict_infos = {
            "hx": hx,
            "hx_next": sol.value(hx_next),
            "cbf_lhs": sol.value(hx_next - hx + slack),
            "cbf_rhs": -gamma * hx,
            "slack": sol.value(slack),
            "gammas": sol.value(gamma),
            "safe_gammas": sol.value(gamma),
            "obj": sol.value(obj),
            "opt_status": 1.0,
        }

    except Exception:
        safe_input = u

        dict_infos = {
            "hx": 0.0,
            "hx_next": 0.0,
            "cbf_lhs": 0.0,
            "cbf_rhs": 0.0,
            "slack": 0.0,
            "gammas": 0.0,
            "safe_gammas": 0.0,
            "obj": 0.0,
            "opt_status": -1.0,
        }

    return safe_input, dict_infos

def main():
    """
    Create video animation of the trajectory of x, y
    """

    env = DoubleIntegratorEnv(
        render_mode=None,
        dt=0.1,
        time_limit=7.5,
        observation_type="state_and_time",
        reward_type="norm_progress",
    )


    trajectories = {}

    for gamma in [0.1, 0.25, 0.5, 1.0]:
        print("gamma", gamma)

        env.reset()
        done = False

        xs, ys = [], []

        while not done:
            x, y, vx, vy = env.state
            action = 2.5 * (env.goal_xy - env.state[:2])
            action, _ = double_integrator_advanced_cbf_project(env, env.state, action, gamma=gamma)

            obs, reward, done, truncated, info = env.step(action)
            frame = env.render()

            # debug
            xs.append(x)
            ys.append(y)

        trajectories[gamma] = (xs, ys)

    # Create a video of the trajectory
    fig, ax = plt.subplots()
    ax.set_xlim(-6, 0)
    ax.set_ylim(-6, 0)
    ax.set_aspect('equal')

    # Create a circle for the obstacle
    ox, oy = env.obst_xy
    circle = plt.Circle((ox, oy), env.obst_r, color='r', alpha=0.5)
    ax.add_artist(circle)

    # Create a line for the trajectory
    lines = []
    for gamma, (xs, ys) in trajectories.items():
        line, = ax.plot([], [], label=f'$\gamma$={gamma}')
        lines.append(line)

    # place legend outside of plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def init():
        for i in range(len(lines)):
            lines[i].set_data([], [])
        return lines

    def animate(i):
        for j, (gamma, (xs, ys)) in enumerate(trajectories.items()):
            lines[j].set_data(xs[:i], ys[:i])
        return lines

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=200, blit=True)
    anim.save(f'trajectory_all.mp4', writer='ffmpeg', dpi=300)

if __name__== "__main__":
    main()