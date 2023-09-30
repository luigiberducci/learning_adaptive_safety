import gymnasium as gym
import numpy as np
from pyclothoids import Clothoid

from gym_envs.multi_agent_env import MultiAgentRaceEnv
from gym_envs.multi_agent_env.common.utils import get_adaptive_lookahead
from gym_envs.multi_agent_env.planners.frenet_planner import (
    ControllableFrenetPlanner,
    FrenetPlanner,
)
from gym_envs.multi_agent_env.planners.lattice_planner import (
    ControllableLatticePlanner,
    sample_lookahead_square,
    sample_traj,
    LatticePlanner,
)


class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""

    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(self.env.action_space, action)


class LocalPathActionWrapper(gym.Wrapper):
    """
    Action space is a 2d vector of the form [d, v],
    where d is the lateral offset from the raceline and v is the velocity gain at the end of a plan.

    The actions are normalized to the range [-1, 1] and then scaled to the actual range of the action space:
        - action 'd' is mapped to the range [min_d, max_d], and
        - action 'v' mapped to [min_v, max_v]
    """

    planner_fns = {
        "lattice": ControllableLatticePlanner,
        "frenet": ControllableFrenetPlanner,
        "auto_lattice": LatticePlanner,
        "auto_frenet": FrenetPlanner,
    }

    def __init__(
        self,
        env: gym.Env,
        planner_type: str,
        planner_params: dict = {"tracker": "pure_pursuit"},
        min_d: float = -1.5,
        max_d: float = 1.5,
        min_v: float = 0.0,
        max_v: float = 1.0,
    ):
        super().__init__(env)
        self.min_d = min_d
        self.max_d = max_d
        self.min_v = min_v
        self.max_v = max_v
        self.agent_id = "ego"
        self.auto_mode = planner_type.startswith(
            "auto"
        )  # auto mode does not expose path planning actions

        assert isinstance(
            env.unwrapped, MultiAgentRaceEnv
        ), "LocalPathActionWrapper only works with MultiAgentRaceEnv"
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), "LocalPathActionWrapper assumes Box action space, use FlattenAction wrapper"
        assert (
            env.action_space.shape[0] >= 2
        ), "LocalPathActionWrapper assumes at least (steering, velocity)"

        if self.auto_mode:
            # auto mode: reduce action space, get rid of the first two actions
            assert (
                env.action_space.shape[0] >= 2
            ), "Auto mode requires at least two actions"
            assert (
                len(env.action_space.shape) == 1
            ), "Auto mode requires 1d action space"
            self.action_space = gym.spaces.Box(
                low=env.action_space.low[2:],
                high=env.action_space.high[2:],
                shape=(env.action_space.shape[0] - 2,),
                dtype=env.action_space.dtype,
            )
        else:
            # instead if not auto mode, we keep same action_space but later override the first two actions
            pass

        self.planner = self.planner_fns[planner_type](
            track=env.track, params=planner_params, agent_id=self.agent_id
        )

    def step(self, action: np.ndarray):
        if not self.auto_mode:
            # set path goal from d, v actions
            dgoal = (action[0] + 1) / 2 * (self.max_d - self.min_d) + self.min_d
            vgain = (action[1] + 1) / 2 * (self.max_v - self.min_v) + self.min_v

            self.planner.set_action(dgoal=dgoal, vgain=vgain)

        # step planner
        planner_action = self.planner.plan(
            observation=self.original_obs, agent_id=self.agent_id
        )

        if self.auto_mode:
            # extend action
            action = np.concatenate(
                [[planner_action["steering"]], [planner_action["velocity"]], action]
            )
        else:
            # override action
            action[0] = planner_action["steering"]
            action[1] = planner_action["velocity"]

        obs, reward, done, truncated, info = super().step(action=action)
        self.original_obs = obs
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        original_obs, info = super().reset(**kwargs)
        self.original_obs = original_obs
        self.planner.reset(**kwargs)
        return original_obs, info


class WaypointActionWrapper(gym.Wrapper):
    """
    Action space is a 2d vector of the form [d, v],
    where d is the lateral offset from the centerline and v is the normalized velocity at the end of a plan.

    The actions are normalized to the range [-1, 1] and then scaled to the actual range of the action space:
        - action 'd' is mapped to the range [min_d, max_d], and
        - action 'v' mapped to [min_v, max_v]
    """

    def __init__(
        self,
        env: MultiAgentRaceEnv,
        tracker_params: dict = {},
        min_d: float = -1.5,
        max_d: float = 1.5,
        min_v: float = 0.0,
        max_v: float = 8.0,
    ):
        super().__init__(env)
        self.min_d = min_d
        self.max_d = max_d
        self.min_v = min_v
        self.max_v = max_v
        self.agent_id = "ego"

        assert isinstance(
            env.unwrapped, MultiAgentRaceEnv
        ), "LocalPathActionWrapper only works with MultiAgentRaceEnv"
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), "LocalPathActionWrapper assumes Box action space, use FlattenAction wrapper"
        assert (
            env.action_space.shape[0] >= 2
        ), "LocalPathActionWrapper assumes at least (steering, velocity)"

        # it does not change the action space, just overrides the first two actions (from [steer, velocity] to [d, v])
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.orig_action_space = env.action_space
        self.action_space.low[0], self.action_space.high[0] = -1, 1
        self.action_space.low[1], self.action_space.high[1] = -1, 1

        self.waypoints = np.stack(
            [
                env.track.centerline.xs,
                env.track.centerline.ys,
                env.track.centerline.velxs,
                env.track.centerline.yaws,
            ],
            axis=1,
        ).astype(np.float32)
        self.last_gpoint = np.zeros(4, dtype=np.float32)

        from gym_envs.multi_agent_env.planners.pure_pursuit import PurePursuitPlanner
        from gym_envs.multi_agent_env.planners.planner import Planner

        self.tracker: Planner = PurePursuitPlanner(
            track=env.track, params=tracker_params, agent_id=self.agent_id
        )

    def step(self, action: np.ndarray):
        dgoal = (action[0] + 1) / 2 * (self.max_d - self.min_d) + self.min_d
        vgoal = (action[1] + 1) / 2 * (self.max_v - self.min_v) + self.min_v

        # clothoid plan
        pose_x, pose_y, pose_theta = self.original_obs[self.agent_id]["pose"]
        velocity = self.original_obs[self.agent_id]["velocity"][0]
        minL, maxL, Lscale = 0.5, 1.5, 8.0

        lookahead = np.array(
            [get_adaptive_lookahead(velocity, minL, maxL, Lscale)]
        ).astype(np.float32)
        lat_goal = np.array([dgoal]).astype(np.float32)
        try:
            gpoint = sample_lookahead_square(
                pose_x,
                pose_y,
                pose_theta,
                velocity,
                self.waypoints,
                lookahead,
                lat_goal,
            )[0]
        except:
            gpoint = (
                self.last_gpoint
            )  # quick fix to avoid crash when njit has approx issue

        clothoid = Clothoid.G1Hermite(
            pose_x, pose_y, pose_theta, gpoint[0], gpoint[1], gpoint[2]
        )
        local_traj = sample_traj(clothoid, 20, gpoint[3])
        local_traj[:, 2] = vgoal

        # tracking
        control = self.tracker.plan(
            observation=self.original_obs, agent_id=self.agent_id, waypoints=local_traj
        )
        action[0] = control["steering"]
        action[1] = control["velocity"]

        obs, reward, done, truncated, info = super().step(action=action)
        self.original_obs = obs
        self.last_gpoint = gpoint
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        original_obs, info = super().reset(**kwargs)
        self.original_obs = original_obs
        self.tracker.reset(**kwargs)
        self.last_gpoint = np.zeros(4, dtype=np.float32)
        return original_obs, info
