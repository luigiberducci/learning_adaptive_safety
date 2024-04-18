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

from example_double_integrator_obstacle import DoubleIntegratorEnv, double_integrator_advanced_cbf_project

mpl.rcParams['text.usetex'] = True
mpl.rcParams["text.latex.preamble"]  = r"\usepackage{cmbright}"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["font.size"] = 14
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.linewidth"] = 1.0
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["legend.fontsize"] = 20

debug = False

def main():
    """
    Create video animation of the trajectory of x, y
    """

    n_obst = 1
    obst_noise = 0.0
    record = True
    render_mode = None #"human"
    env = DoubleIntegratorEnv(
        n_obstacles=n_obst,
        obst_noise=obst_noise,
        render_mode=render_mode,
        dt=0.05,
        time_limit=1.3,
        observation_type="state_and_time",
        reward_type="norm_progress",
    )


    trajectories = {}
    trajectories_hx = {}
    obst_trajectories = {}
    actions = {}

    for gamma in [0.8, 0.2]:
        print("gamma", gamma)

        env.reset()
        done = False

        xs, ys = [], []
        oxs, oys = [], []
        acts = []
        hxs = []

        while not done:
            x, y, vx, vy = env.state
            action = 5 * np.array([-x, -y])
            action, action_info = double_integrator_advanced_cbf_project(env, env.state, action, gamma=gamma)

            obs, reward, done, truncated, info = env.step(action)
            frame = env.render()

            # debug
            xs.append(x)
            ys.append(y)
            oxs.append(env.obst_xy[:, 0])
            oys.append(env.obst_xy[:, 1])
            hxs.append(min(action_info["hxs"]))
            acts.append(action)

        trajectories[gamma] = (xs, ys)
        trajectories_hx[gamma] = hxs
        obst_trajectories[gamma] = (oxs, oys)
        actions[gamma] = acts

    # Create a video of the trajectory
    if not record:
        exit(0)


    fig = plt.figure(figsize=(5, 9))
    axtraj = plt.subplot(211)
    axh = plt.subplot(413)
    fig.tight_layout()

    axtraj.set_xlim(-6, 0)
    axtraj.set_ylim(-6, 0)
    axtraj.set_aspect('equal')
    axtraj.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    axtraj.set_xticks([])
    axtraj.set_yticks([])


    axh.set_xlim(0, len(xs))
    axh.set_ylim(0.0, 3.0)
    axh.set_xlabel("Time")
    axh.set_ylabel("$h(s_t)$")
    axh.spines[['right', 'top']].set_visible(False)


    # Create a circle for the obstacles
    circles = []
    for ox, oy in env.obst_xy:
        circle = plt.Circle((ox, oy), env.obst_r, color='r', alpha=0.30)
        circles.append(circle)

    # Create a line for the trajectory
    lines = []
    lines_hx = []
    for gamma, _ in trajectories.items():
        line, = axtraj.plot([], [], label=f'$\gamma$={gamma}', marker="o", markersize=3)
        lines.append(line)

        line_h, = axh.plot([], [], label=f'$\gamma$={gamma}', marker="o", markersize=2)
        lines_hx.append(line_h)


    # place legend outside of plot
    axh.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)

    def init():
        for i in range(len(lines)):
            lines[i].set_data([], [])
        for i in range(len(circles)):
            circles[i].center = (env.obst_xy[i, 0], env.obst_xy[i, 1])
            axtraj.add_patch(circles[i])

        for i in range(len(lines_hx)):
            lines_hx[i].set_data([], [])

        return lines + circles + lines_hx

    def animate(i):
        for j, (gamma, (xs, ys)) in enumerate(trajectories.items()):
            lines[j].set_data(xs[:i], ys[:i])
        for j, (gamma, (oxs, oys)) in enumerate(obst_trajectories.items()):
            for k in range(len(oxs[0])):
                circles[k].center = (oxs[i][k], oys[i][k])
        for j, (gamma, hx) in enumerate(trajectories_hx.items()):
            lines_hx[j].set_data(np.arange(i), hx[:i])
        return lines + circles + lines_hx

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=300, blit=True)
    anim.save(f'trajectory_two_gamma.mp4', writer='ffmpeg', dpi=300)

if __name__== "__main__":
    main()