import re
from abc import abstractmethod
from typing import List, Tuple, Union

import numpy as np

from gym_envs.multi_agent_env.common.track import Track
import gymnasium as gym

Pose = Union[Tuple[float, float, float], np.ndarray]

DEBUG_RESET = False


class ResetFn:
    @abstractmethod
    def sample(self) -> List[Pose]:
        pass


class SectionMaskFn:
    @abstractmethod
    def __call__(self) -> np.ndarray:
        pass


class PositioningFn:
    @abstractmethod
    def __call__(self, **kwargs) -> List[Pose]:
        pass


"""
Section masking functions
"""


class GridMaskFn(SectionMaskFn):
    """
    Return a mask of the waypoints close to the starting line.
    """

    def __init__(self, track: Track, starting_line_width: float = 1.0):
        self.track = track
        self.starting_line_width = starting_line_width

        step_size = self.track.centerline.length / self.track.centerline.n
        self.n_wps = int(self.starting_line_width / step_size)

        mask = np.zeros(self.track.centerline.n)
        mask[: self.n_wps] = 1
        self.mask = mask.astype(bool)

    def __call__(self) -> np.ndarray:
        return self.mask


class SpecSectionMaskFn(SectionMaskFn):
    """
    Return a mask of the waypoints close to the starting line.
    """

    def __init__(
        self, track: Track, min_section: float = 0.0, max_section: float = 0.0
    ):
        assert (
            min_section >= 0.0 and min_section <= 1.0
        ), "section_fraction must be in [0, 1]"
        assert (
            max_section >= 0.0 and max_section <= 1.0
        ), "section_fraction must be in [0, 1]"
        assert (
            min_section <= max_section
        ), "min_section must be smaller than max_section"
        self.track = track
        self.min_section = min_section
        self.max_section = max_section

        begin = int((self.track.centerline.n - 1) * self.min_section)
        end = int((self.track.centerline.n - 1) * self.max_section)

        mask = np.zeros(self.track.centerline.n)
        mask[begin : end + 1] = 1
        self.mask = mask.astype(bool)

    def __call__(self) -> np.ndarray:
        return self.mask


class StaticMaskFn(SectionMaskFn):
    """
    Return a mask of first waypoint, to make the execution as deterministic as possible.
    """

    def __init__(self, track: Track):
        self.track = track

        mask = np.zeros(self.track.centerline.n)
        mask[1] = 1
        self.mask = mask.astype(bool)

    def __call__(self) -> np.ndarray:
        return self.mask


class AllTrackMaskFn(SectionMaskFn):
    """
    Return a mask of all waypoints.
    """

    def __init__(self, track: Track):
        self.track = track
        self.mask = np.ones(self.track.centerline.n).astype(bool)

    def __call__(self) -> np.ndarray:
        return self.mask


class CurvatureConstrainedMaskFn(SectionMaskFn):
    """
    Return a mask of the waypoints in proximity of sections with curv within limits.
    """

    def __init__(
        self,
        track: Track,
        min_curvature: float = np.NINF,
        max_curvature: float = np.PINF,
        length: float = 5,
    ):
        self.track = track
        self.section_length = length
        self.min_curvature = min_curvature
        self.max_curvature = max_curvature
        self.filter = "avg"

        # filter curvature
        abs_ks = np.abs(self.track.centerline.kappas)
        curve = self._filter(array=abs_ks, type=self.filter, w_len=50)
        curve = (curve - curve.min()) / (curve.max() - curve.min())

        # compute lookahead max curvature
        centerline_step = (
            self.track.centerline.length / self.track.centerline.n
        )  # meters
        n_wps_x_section = int(
            self.section_length / centerline_step
        )  # approx nr waypoints in a section
        max_section_curv = np.asarray(
            [np.max(curve[i : i + n_wps_x_section]) for i in range(len(curve))]
        )

        # section selection
        mask_curve = np.logical_and(
            max_section_curv >= self.min_curvature,
            max_section_curv <= self.max_curvature,
        )
        masked_curve = np.where(mask_curve, np.gradient(curve), 1000)

        self.mask = masked_curve < 0.1

        if not any(self.mask):
            raise ValueError(
                f"No valid waypoints in {track.spec.name} with the given curvature constraints. Check centerline."
            )

    def __call__(self) -> np.ndarray:
        return self.mask

    @staticmethod
    def _filter(array: np.ndarray, w_len: int, type: str = None) -> np.ndarray:
        if type is None:
            return array
        elif type == "avg":
            fn = np.mean
        elif type == "median":
            fn = np.median
        else:
            raise ValueError(f"Unknown filter type {type}")
        return np.array(
            [
                fn(array[i1:i2])
                for i1, i2 in zip(range(len(array) - w_len), range(w_len, len(array)))
            ]
        )


"""
Positioning functions
"""


class PositionAroundWaypoint(PositioningFn):
    def __init__(self, track: Track, min_dist: float = None, max_dist: float = None):
        self.track = track
        self.min_dist = min_dist
        self.max_dist = max_dist

        # by default, choose min/max dist based on track length
        auto_min_dist, auto_max_dist = self.get_auto_sampling_range(
            track.centerline.length
        )
        self.min_dist = (
            auto_min_dist if min_dist is None else min_dist
        )  # if user-provided, use it, otherwise use the auto-computed value
        self.max_dist = auto_max_dist if max_dist is None else max_dist

        # sanity check
        assert self.min_dist <= self.max_dist, "min_dist must be smaller than max_dist"
        assert self.min_dist >= 0, f"min_dist must be positive, got {min_dist} instead"
        assert self.max_dist >= 0, f"max_dist must be positive, got {max_dist} instead"

    def __call__(
        self, waypoint_id: int, n_agents: int, shuffle: bool = True
    ) -> List[Pose]:
        """
        Compute agents poses by iteratively sampling the next agent within a distance range from the previous one.

        :param n_agents: the number of agents
        :param waypoint_id: the id of the first waypoint from which start the sampling
        :param min_dist: the minimum distance between two consecutive agents
        :param max_dist: the maximum distance between two consecutive agents
        """
        current_wp_id = waypoint_id
        n_waypoints = self.track.centerline.n

        poses = []
        rnd_sign = (
            np.random.choice([-1, 1]) if shuffle else 1
        )  # random sign to sample lateral position (left/right)
        for i in range(n_agents):
            # compute pose from current wp_id
            wp = [
                self.track.centerline.xs[current_wp_id],
                self.track.centerline.ys[current_wp_id],
            ]
            next_wp_id = (current_wp_id + 1) % n_waypoints
            next_wp = [
                self.track.centerline.xs[next_wp_id],
                self.track.centerline.ys[next_wp_id],
            ]
            theta = np.arctan2(next_wp[1] - wp[1], next_wp[0] - wp[0])

            x, y = wp[0], wp[1]
            if n_agents > 1:
                lat_offset = rnd_sign * (-1.0) ** i * (1.0 / n_agents)
                x += lat_offset * np.cos(theta + np.pi / 2)
                y += lat_offset * np.sin(theta + np.pi / 2)

            pose = np.array([x, y, theta])
            poses.append(pose)
            # find id of next waypoint which has mind <= dist <= maxd
            first_id, interval_len = (
                None,
                None,
            )  # first wp id with dist > mind, len of the interval btw first/last wp
            pnt_id = current_wp_id  # moving pointer to scan the next waypoints
            dist = 0.0
            while dist <= self.max_dist:
                # sanity check
                if pnt_id > n_waypoints - 1:
                    pnt_id = 0
                # increment distance
                x_diff = (
                    self.track.centerline.xs[pnt_id]
                    - self.track.centerline.xs[pnt_id - 1]
                )
                y_diff = (
                    self.track.centerline.ys[pnt_id]
                    - self.track.centerline.ys[pnt_id - 1]
                )
                dist = dist + np.linalg.norm(
                    [y_diff, x_diff]
                )  # approx distance by summing linear segments
                # look for sampling interval
                if first_id is None and dist >= self.min_dist:  # not found first id yet
                    first_id = pnt_id
                    interval_len = 0
                if (
                    first_id is not None and dist <= self.max_dist
                ):  # found first id, increment interval length
                    interval_len += 1
                pnt_id += 1
            # sample next waypoint
            current_wp_id = (first_id + np.random.randint(0, interval_len + 1)) % (
                n_waypoints
            )

        return poses

    @staticmethod
    def get_auto_sampling_range(track_length: float) -> Tuple[float, float]:
        """
        Compute the default sampling range based on the track length.

        :param track_length: length of the track in meters
        :return: min, max starting distance between agents
        """
        min_dist, max_dist = (
            1.5,
            2.0,
        )  # default for small tracks, with less than 10m centerline
        if track_length > 15:
            min_dist *= 2  # double the range for large-enough tracks
            max_dist *= 2
        return min_dist, max_dist


class PositionEgoFront(PositionAroundWaypoint):
    def __call__(self, *args, **kwargs):
        poses = super().__call__(*args, **kwargs)
        poses = poses[::-1]
        return poses


class PositionEgoBack(PositionAroundWaypoint):
    def __call__(self, *args, **kwargs):
        poses = super().__call__(*args, **kwargs)
        return poses


class PositionStaticEgoBack(PositionAroundWaypoint):
    def __call__(self, *args, **kwargs):
        poses = super().__call__(*args, **kwargs, shuffle=False)
        return poses


class PositionH2H(PositionAroundWaypoint):
    def __init__(self, track: Track):
        min_dist = max_dist = 0.0
        super().__init__(track=track, min_dist=min_dist, max_dist=max_dist)

    def __call__(self, *args, **kwargs):
        poses = super().__call__(*args, **kwargs)
        return poses


class PositionRandom(PositionAroundWaypoint):
    def __call__(self, *args, **kwargs):
        poses = super().__call__(*args, **kwargs)
        # shuffle
        np.random.shuffle(poses)
        return poses


class SectionPositioningResetFn(ResetFn):
    def __init__(self, env: gym.Env, section_mode: str, relpos_mode: str):
        assert hasattr(env, "track") and isinstance(
            env.track, Track
        ), "a racing env must have a track attribute"
        self.waypoint_masking = section_mask_factory(
            section_mode, track=env.track, num_agents=env.num_agents
        )
        self.positioning_fn = positioning_factory(
            relpos_mode, track=env.track, num_agents=env.num_agents
        )
        self.n_agents = env.num_agents if hasattr(env, "num_agents") else 1

    def sample(self) -> List[Pose]:
        valid_waypoints = self.waypoint_masking()
        waypoint_id = np.random.choice(np.where(valid_waypoints)[0])
        poses = self.positioning_fn(waypoint_id=waypoint_id, n_agents=self.n_agents)

        if DEBUG_RESET:
            print(f"[debug] poses: {poses}")

        return poses


SECTION_MODES = [
    "grid",
    "random",
    "section",
]  # "straight", "softcurve", "hardcurve", "static"]
RELPOS_MODES = ["front", "back", "h2h", "random"]  # "static", "static2"]
RESET_MODES = [f"{s}_{r}" for s in SECTION_MODES for r in RELPOS_MODES]


def section_mask_factory(
    section_mode_str: str, track: Track, num_agents: int
) -> SectionMaskFn:
    section_mode = re.sub(r"\d+\.\d+(-\d+\.\d+)?", "", section_mode_str)
    section_frac = re.findall("\d+\.\d+", section_mode_str)
    assert section_mode in SECTION_MODES, f"section_mode must be in {SECTION_MODES}"

    if section_mode == "grid":
        return GridMaskFn(track=track)
    elif section_mode == "section":
        assert len(section_frac) in [
            0,
            1,
            2,
        ], "section mode must be in format 'section_<frac>' or 'section_<frac1>-<frac2>'"
        min_section = float(section_frac[0]) if len(section_frac) > 0 else 0.0
        max_section = float(section_frac[1]) if len(section_frac) == 2 else min_section
        return SpecSectionMaskFn(
            track=track, min_section=min_section, max_section=max_section
        )
    elif section_mode == "static":
        return StaticMaskFn(track=track)
    elif section_mode == "random":
        return AllTrackMaskFn(track=track)
    elif section_mode == "straight":
        return CurvatureConstrainedMaskFn(
            track=track, min_curvature=0.0, max_curvature=0.05
        )
    elif section_mode == "softcurve":
        return CurvatureConstrainedMaskFn(
            track=track, min_curvature=0.05, max_curvature=0.20
        )
    elif section_mode == "hardcurve":
        return CurvatureConstrainedMaskFn(track=track, min_curvature=0.20)
    else:
        raise ValueError(f"Invalid section_mode: {section_mode}")


def positioning_factory(
    positioning_str: str, track: Track, num_agents: int
) -> PositioningFn:
    positioning_mode = re.sub(r"\d+.?\d+", "", positioning_str)
    distance = re.findall("\d+\.?\d+", positioning_str)
    assert (
        positioning_mode in RELPOS_MODES
    ), f"positioning_mode must be in {RELPOS_MODES}"

    if positioning_mode == "front":
        dist = float(distance[0]) if len(distance) > 0 else None
        return PositionEgoFront(track=track, min_dist=dist, max_dist=dist)
    elif positioning_mode == "back":
        dist = float(distance[0]) if len(distance) > 0 else None
        return PositionEgoBack(track=track, min_dist=dist, max_dist=dist)
    elif positioning_mode == "static":
        return PositionStaticEgoBack(track=track, min_dist=1.5, max_dist=1.5)
    elif positioning_mode == "static2":
        return PositionStaticEgoBack(track=track, min_dist=3.0, max_dist=3.0)
    elif positioning_mode == "h2h":
        return PositionH2H(track=track)
    elif positioning_mode == "random":
        dist = float(distance[0]) if len(distance) > 0 else None
        return PositionRandom(track=track, min_dist=dist, max_dist=dist)
    else:
        raise ValueError(f"Invalid positioning_mode: {positioning_mode}")


def reset_fn_factory(env: gym.Env, reset_type: str) -> ResetFn:
    tokens = reset_type.split("_")
    assert (
        len(tokens) == 2
    ), f"reset_type must be in the form '<section_mode>_<relpos_mode>', got {reset_type}"
    section_mode, positioning_mode = tokens[0], tokens[1]
    return SectionPositioningResetFn(
        env=env, section_mode=section_mode, relpos_mode=positioning_mode
    )
