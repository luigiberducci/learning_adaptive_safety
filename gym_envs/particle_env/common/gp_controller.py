import pathlib
from typing import List, Union, Tuple, Dict

import numpy as np
from pydantic.utils import deep_update

from gym_envs.particle_env.common.GP_predict import GP
from gym_envs.particle_env.common.car import Car
from gym_envs.particle_env.common.control import (
    get_trajectory,
    filter_output,
    filter_output_decay,
    filter_output_primal_decay,
    filter_output_primal,
)

data_dir = pathlib.Path(__file__).parent.parent.absolute() / "data"


class CBFCar(Car):
    def __init__(self, x: float, y: float, N_a: int, params={}):
        super().__init__(x, y)

        self.params = {
            "use_clf": True,
            "use_cbf": True,
            "robust": True,
            "opt_decay": False,
            "odcbf_gamma_penalty": 1e5,  # penalty term for deviation from nominal gamma
            "eps": 1e-9,
            "p_threshold": 1 - 0.95,
            "l": 60.0,
            "sigma": 8.5,
            "noise": 0.01,
            "horizon": 15,
        }
        self.params = deep_update(self.params, params)

        # internal variables
        self.N_a = N_a

        # gp
        self.all_gp = None
        self.G_all = None
        self.g_all = None
        self.m_all = None
        self.z_all = None
        self.G_base = None
        self.g_base = None

        self.reset()

    def reset(self):
        if not self.params["robust"]:
            return

        N_a = self.N_a
        eps = self.params["eps"]
        l = self.params["l"]
        sigma = self.params["sigma"]
        noise = self.params["noise"]
        horizon = self.params["horizon"]

        self.all_gp = []

        gp = GP(
            X=None,
            Y=None,
            omega=np.eye(4),
            l=l,
            sigma=sigma,
            noise=noise,
            horizon=horizon,
        )
        self.all_gp.append(gp)  # ego GP
        self.all_gp[0].load_parameters(f"{data_dir}/hyperparameters_robot.pkl")

        for i in range(1, N_a):
            gp = GP(
                X=None,
                Y=None,
                omega=np.eye(4),
                l=l,
                sigma=sigma,
                noise=noise,
                horizon=horizon,
            )
            self.all_gp.append(gp)  # Human GP
            self.all_gp[i].load_parameters(f"{data_dir}/hyperparameters_human.pkl")

        self.G_all = np.zeros((N_a - 1, 8 * 2, 8))
        self.g_all = np.zeros((N_a - 1, 8 * 2))
        self.m_all = np.zeros((N_a, 4))
        self.z_all = np.zeros((N_a, 2))
        self.G_base = np.array(
            [
                [1, 0, 0, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, -1],
            ]
        )

        self.g_base = np.array([eps, eps, eps, eps, eps, eps, eps, eps])

    def plan(
        self,
        state: np.ndarray,
        d_state: np.ndarray,
        agents: List[Car],
        gamma: Union[float, np.ndarray],
        goal: np.ndarray = None,
    ) -> np.ndarray:
        # if use clf, go to the goal point, ignoring any subgoal given
        if self.params["use_clf"]:
            goal = None

        try:
            u, x_path, x0 = get_trajectory(self, N=10, goal=goal)
            opt_info = {"opt_status": 1.0}
        except ValueError as e:
            u = np.zeros(2)
            opt_info = {"opt_status": -1.0}
            return u, opt_info

        # safeguarding with CBF
        if not self.params["use_cbf"]:
            # no CBF
            return u, opt_info
        elif self.params["robust"]:
            # robust CBF
            u = self.robust_project(state, x_path, d_state, agents, gamma)
        else:
            # simple CBF
            u = self.cbf_project(x_path, agents, gamma)

        return u, opt_info

    def robust_project(
        self,
        state: np.ndarray,
        x_path: np.ndarray,
        d_state: np.ndarray,
        agents: List[Car],
        gamma: Union[float, np.ndarray],
    ):
        eps = self.params["eps"]
        p_threshold = self.params["p_threshold"]

        # update GP
        self.all_gp[0].add_data(state[0, :], d_state[0, :])
        self.all_gp[0].get_obs_covariance()
        for j in range(1, self.N_a):
            self.all_gp[j].add_data(state[j, :], d_state[j, :])
            self.all_gp[j].get_obs_covariance()

        # Infer uncertainty polytope for robot and other agents
        if self.all_gp[0].N_data > 0:
            m_d, cov_d = self.all_gp[0].predict(state[0, :])
            G_r, g_r = self.all_gp[0].extract_box(
                cov_d, p_threshold=p_threshold
            )  # G: 8x4, g: 8x1
            self.m_all[0, :] = m_d
            self.z_all[0, 0], self.z_all[0, 1] = self.all_gp[0].extract_norms(
                cov_d, p_threshold=p_threshold
            )

        for j in range(1, self.N_a):
            if self.all_gp[j].N_data > 0:
                m_d, cov_d = self.all_gp[j].predict(state[j, :])
                G_h, g_h = self.all_gp[j].extract_box(cov_d, p_threshold=p_threshold)
                self.m_all[j, :] = m_d
                self.z_all[j, 0], self.z_all[j, 1] = self.all_gp[j].extract_norms(
                    cov_d, p_threshold=p_threshold
                )
                self.G_all[j - 1, 0:8, 0:4] = G_r
                self.g_all[j - 1, 0:8] = g_r
                if np.linalg.norm(state[j, 2:4]) < eps:
                    self.G_all[j - 1, 8:16, 4:8] = self.G_base
                    self.g_all[j - 1, 8:16] = self.g_base
                else:
                    self.G_all[j - 1, 8:16, 4:8] = G_h
                    self.g_all[j - 1, 8:16] = g_h

        # Obtain safe control given uncertainty polytopes
        if self.all_gp[0].N_data > 0:
            if self.params["opt_decay"]:
                u = filter_output_decay(
                    0,
                    agents,
                    x_path,
                    G_all=self.G_all,
                    g_all=self.g_all,
                    m=self.m_all,
                    z=self.z_all,
                    gamma=gamma,
                    gamma_penalize=self.params["odcbf_gamma_penalty"],
                )
            else:
                u = filter_output(
                    0,
                    agents,
                    x_path,
                    G_all=self.G_all,
                    g_all=self.g_all,
                    m=self.m_all,
                    z=self.z_all,
                    gamma=gamma,
                )
        else:
            u = self.cbf_project(x_path, agents, gamma)

        return u

    def cbf_project(
        self,
        x_path: np.ndarray,
        agents: List[Car],
        gamma: Union[float, np.ndarray],
    ):
        if self.params["opt_decay"]:
            u, _ = filter_output_primal_decay(
                0,
                agents,
                x_path,
                gamma=gamma,
                gamma_penalize=self.params["odcbf_gamma_penalty"],
            )
        else:
            u, _ = filter_output_primal(0, agents, x_path, gamma=gamma)
        return u
