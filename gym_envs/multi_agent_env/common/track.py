import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose
from numba import njit
from yamldataclassconfig.config import YamlDataClassConfig

from gym_envs.multi_agent_env.common.splines.cubic_spline import CubicSpline2D
from gym_envs.multi_agent_env.common.utils import nearest_point


@dataclass
class TrackSpec(YamlDataClassConfig):
    name: str
    image: str
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float


class Raceline:
    n: int

    ss: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    yaws: np.ndarray
    kappas: np.ndarray
    velxs: np.ndarray
    accxs: np.ndarray

    spline: CubicSpline2D
    length: float

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        velxs: np.ndarray,
        drights: np.ndarray = None,
        dlefts: np.ndarray = None,
        ss: np.ndarray = None,
        psis: np.ndarray = None,
        kappas: np.ndarray = None,
        accxs: np.ndarray = None,
        spline: CubicSpline2D = None,
    ):
        assert xs.shape == ys.shape == velxs.shape, "inconsistent shapes for x, y, vel"

        self.n = xs.shape[0]
        self.ss = ss
        self.xs = xs
        self.ys = ys
        self.yaws = psis
        self.kappas = kappas
        self.velxs = velxs
        self.accxs = accxs
        self.drights = drights
        self.dlefts = dlefts


        # approximate track length by linear-interpolation of x,y waypoints
        # note: we could use 'ss' but sometimes it is normalized to [0,1], so we recompute it here
        x_diffs = [nx - x for x, nx in zip(self.xs[:-1], self.xs[1:])]
        y_diffs = [ny - y for y, ny in zip(self.ys[:-1], self.ys[1:])]
        self.length = np.linalg.norm([y_diffs, x_diffs], axis=0).sum()

        self.spline = spline if spline is not None else CubicSpline2D(xs, ys)

    @staticmethod
    def from_centerline_file(
        filepath: pathlib.Path, delimiter: str = ",", fixed_speed: float = 1.0
    ):
        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        waypoints = np.loadtxt(filepath, delimiter=delimiter)
        assert waypoints.shape[1] == 4, "expected waypoints as [x, y, w_left, w_right]"

        # fit cubic spline to waypoints
        step = (
            10 if "General1" in filepath.name else 1
        )  # General1 has very bad centerline, need to be smoothed
        xx, yy = waypoints[::step, 0], waypoints[::step, 1]

        # close loop
        xx = np.append(xx, xx[0])
        yy = np.append(yy, yy[0])
        spline = CubicSpline2D(xx, yy)
        ds = 0.1

        ss, xs, ys, yaws, ks, ddr, ddl = [], [], [], [], [], [], []
        smoothyaws, smoothks = [], []

        for i_s in np.arange(0, spline.s[-1], ds):
            x, y = spline.calc_position(i_s)
            pos = np.array([x, y])
            _, _, _, i = nearest_point(pos, waypoints[:, :2])
            i = i % waypoints.shape[0]
            dr = waypoints[i, 2]
            dl = waypoints[i, 3]

            yaw = spline.calc_yaw(i_s)
            k = spline.calc_curvature(i_s)

            syaw = spline.calc_smooth_yaw(i_s)
            sk = spline.calc_smooth_curvature(i_s)

            xs.append(x)
            ys.append(y)
            ddr.append(dr)
            ddl.append(dl)
            yaws.append(yaw)
            ks.append(k)
            ss.append(i_s)
            smoothks.append(sk)
            smoothyaws.append(syaw)

        return Raceline(
            ss=np.array(ss),
            xs=np.array(xs),
            ys=np.array(ys),
            velxs=np.ones_like(ss) * fixed_speed,
            drights=np.array(ddr),
            dlefts=np.array(ddl),
            psis=np.array(smoothyaws),
            kappas=np.array(smoothks),
            spline=spline,
        )

    @staticmethod
    def from_raceline_file(filepath: pathlib.Path, delimiter: str = ";"):
        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        waypoints = np.loadtxt(filepath, delimiter=delimiter)
        assert (
            waypoints.shape[1] == 7
        ), "expected waypoints as [s, x, y, psi, k, vx, ax]"
        return Raceline(
            ss=waypoints[:, 0],
            xs=waypoints[:, 1],
            ys=waypoints[:, 2],
            psis=waypoints[:, 3],
            kappas=waypoints[:, 4],
            velxs=waypoints[:, 5],
            accxs=waypoints[:, 6],
        )

    @staticmethod
    def from_our_raceline_file(filepath: pathlib.Path, delimiter: str = ","):
        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        waypoints = np.loadtxt(filepath, delimiter=delimiter)
        assert waypoints.shape[1] == 5, "expected waypoints as [x, y, vx, psi, k]"
        return Raceline(
            xs=waypoints[:, 0],
            ys=waypoints[:, 1],
            velxs=waypoints[:, 2],
            psis=waypoints[:, 3],
            kappas=waypoints[:, 4],
        )


def find_track_dir(track_name):
    # we assume there are no blank space in the track name. however, to take into account eventual blank spaces in
    # the map dirpath, we loop over all possible maps and check if there is a matching with the current track
    overrides_dir = (
        pathlib.Path(__file__).parent.parent.parent.parent
        / "f1tenth_racetracks_overrides"
    )
    backup_dir = (
        pathlib.Path(__file__).parent.parent.parent.parent / "f1tenth_racetracks"
    )

    for base_dir in [overrides_dir, backup_dir]:
        if not base_dir.exists():
            continue

        for dir in base_dir.iterdir():
            if track_name == str(dir.stem).replace(" ", ""):
                return dir

    raise FileNotFoundError(
        f"no mapdir matching {track_name} in {[overrides_dir, backup_dir]}"
    )


@dataclass
class Track:
    spec: TrackSpec
    filepath: str
    ext: str
    occupancy_map: np.ndarray
    centerline: Raceline
    raceline: Raceline

    def __init__(
        self,
        spec: TrackSpec,
        filepath: str,
        ext: str,
        occupancy_map: np.ndarray,
        centerline: Raceline,
        raceline: Raceline,
    ):
        self.spec = spec
        self.filepath = filepath
        self.ext = ext
        self.occupancy_map = occupancy_map
        self.centerline = centerline
        self.raceline = raceline
        # approximate track length by linear-interpolation of centerline waypoints
        self.track_length = self.centerline.length
        self.track_width = 3.0

    def get_id_closest_point2centerline(
        self, point: Tuple[float, float], min_id: int = 0
    ):
        idx = (np.linalg.norm(self.centerline[min_id:, 0:2] - point, axis=1)).argmin()
        return idx

    def get_progress(
        self,
        point: Tuple[float, float],
        above_val: float = 0.0,
        return_meters: bool = False,
    ):
        """get progress by looking the closest waypoint with at least `above_val` progress"""
        assert (
            0 <= above_val <= 1
        ), f"progress must be in 0,1 (instead given above_val={above_val})"
        n_points = self.centerline.n
        min_id = int(above_val * n_points)
        idx = self.get_id_closest_point2centerline(point, min_id=min_id)
        progress = idx / n_points
        assert 0 <= progress <= 1, f"progress out of bound {progress}"
        if return_meters:
            progress *= self.track_length
        return progress

    @staticmethod
    def from_track_name(track: str):
        try:
            track_dir = find_track_dir(track)
            # load track spec
            with open(track_dir / f"{track}_map.yaml", "r") as yaml_stream:
                map_metadata = yaml.safe_load(yaml_stream)
                track_spec = TrackSpec(name=track, **map_metadata)

            # load occupancy grid
            map_filename = pathlib.Path(track_spec.image)
            image = Image.open(track_dir / str(map_filename)).transpose(
                Transpose.FLIP_TOP_BOTTOM
            )
            occupancy_map = np.array(image).astype(np.float32)
            occupancy_map[occupancy_map <= 128] = 0.0
            occupancy_map[occupancy_map > 128] = 255.0

            # load centerline and raceline
            centerline = Raceline.from_centerline_file(
                track_dir / f"{track}_centerline.csv"
            )
            if (track_dir / f"{track}_raceline.csv").exists():
                raceline = Raceline.from_raceline_file(
                    track_dir / f"{track}_raceline.csv"
                )
            else:
                raceline = centerline

            return Track(
                spec=track_spec,
                filepath=str((track_dir / map_filename.stem).absolute()),
                ext=map_filename.suffix,
                occupancy_map=occupancy_map,
                centerline=centerline,
                raceline=raceline,
            )

        except Exception as ex:
            print(f"map {track} not found\n{ex}")
            exit(-1)


@njit(cache=True)
def extract_forward_curvature(
    trajectory_xyk: np.ndarray,
    point: np.ndarray,
    forward_distance: float = 10.0,
    n_points: int = 10,
) -> np.ndarray:
    """extract forward curvature starting a given point on the track"""
    _, _, _, begin_i = nearest_point(point, trajectory_xyk[:, :2])

    dist = 0.0
    final_waypoint_id = begin_i
    while dist < forward_distance:
        final_waypoint_id = (final_waypoint_id + 1) % trajectory_xyk.shape[0]
        dist += np.linalg.norm(
            trajectory_xyk[final_waypoint_id, :2]
            - trajectory_xyk[final_waypoint_id - 1, :2]
        )

    ks = np.empty(n_points)
    for i in range(n_points):
        idx = (begin_i + i) % trajectory_xyk.shape[0]
        ks[i] = trajectory_xyk[idx, 2]

    return ks


@njit(cache=True)
def extract_forward_raceline(
    trajectory_xyv: np.ndarray,
    point: np.ndarray,
    forward_distance: float = 10.0,
    n_points: int = 10,
) -> np.ndarray:
    """extract forward curvature starting a given point on the track"""
    _, _, _, begin_i = nearest_point(point, trajectory_xyv[:, :2])

    dist = 0.0
    final_waypoint_id = begin_i
    while dist < forward_distance:
        final_waypoint_id = (final_waypoint_id + 1) % trajectory_xyv.shape[0]
        dist += np.linalg.norm(
            trajectory_xyv[final_waypoint_id, :2]
            - trajectory_xyv[final_waypoint_id - 1, :2]
        )

    forward_xyvs = np.empty((n_points, 3))
    for i in range(n_points):
        idx = (begin_i + i) % trajectory_xyv.shape[0]
        forward_xyvs[i] = trajectory_xyv[idx]

    return forward_xyvs


if __name__ == "__main__":
    track = Track.from_track_name("MexicoCity")
    print("[Result] map loaded successfully")
