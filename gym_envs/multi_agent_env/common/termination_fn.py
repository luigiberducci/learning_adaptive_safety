from abc import abstractmethod


class TerminationFn:
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, state: dict, info: dict = {}) -> bool:
        pass

    def reset(self):
        pass


class TerminateOnEgoCollisionFn(TerminationFn):
    def __call__(self, state: dict, info: dict = {}) -> bool:
        assert "ego" in state, "state must have an ego key"
        return state["ego"]["collision"] > 0


class TerminateOnAnyCollisionFn(TerminationFn):
    def __call__(self, state: dict, info: dict = {}) -> bool:
        agent_ids = list(state.keys())
        return any([state[agent_id]["collision"] > 0 for agent_id in agent_ids])


class TerminateOnAllCompleteNumberLaps(TerminationFn):
    def __init__(self, n_laps: int = 1, **kwargs):
        self.n_laps = n_laps

    def __call__(self, state: dict, info: dict = {}) -> bool:
        assert "lap_counts" in info, "info must have a lap_counts of all agents"
        return all(
            info["lap_counts"] > self.n_laps - 1
        )  # account for starting lap-counter from 0


class TerminateOnAnyCompleteNumberLaps(TerminationFn):
    def __init__(self, n_laps: int = 1, **kwargs):
        self.n_laps = n_laps

    def __call__(self, state: dict, info: dict = {}) -> bool:
        assert "lap_counts" in info, "info must have a lap_counts of all agents"
        return any(
            info["lap_counts"] > self.n_laps - 1
        )  # account for starting lap-counter from 0


class TerminateOnAnyCrossFinishLine(TerminationFn):
    def __call__(self, state: dict, info: dict = {}) -> bool:
        agent_ids = list(state.keys())
        finish_line_s = (
            info["track_length"] - 1.0
        )  # finish line is at the end of the track, 1 meter before the end
        return any(
            [
                state[agent_id]["frenet_coords"][0] > finish_line_s
                for agent_id in agent_ids
            ]
        )


class TerminateOnTimeout(TerminationFn):
    def __init__(self, timeout: float = 1.0, **kwargs):
        self.timeout = timeout

    def __call__(self, state: dict, info: dict = {}) -> bool:
        assert "ego" in state and "time" in state["ego"], "expected time in ego state"
        return bool(state["ego"]["time"] >= self.timeout)


TERMINATION_MODES = [
    "on_collision",
    "on_any_collision",
    "on_all_complete_lap",
    "on_any_complete_lap",
    "on_any_cross_finish_line",
    "on_timeout",
]


def termination_fn_factory(termination_type: str, **kwargs) -> TerminationFn:
    if termination_type == "on_collision":
        return TerminateOnEgoCollisionFn(**kwargs)
    elif termination_type == "on_any_collision":
        return TerminateOnAnyCollisionFn(**kwargs)
    elif termination_type == "on_all_complete_lap":
        return TerminateOnAllCompleteNumberLaps(**kwargs)
    elif termination_type == "on_any_complete_lap":
        return TerminateOnAnyCompleteNumberLaps(**kwargs)
    elif termination_type == "on_any_cross_finish_line":
        return TerminateOnAnyCrossFinishLine(**kwargs)
    elif termination_type == "on_timeout":
        return TerminateOnTimeout(**kwargs)
    else:
        raise ValueError(
            "termination type {termination_type} not in {TERMINATION_MODES}"
        )
