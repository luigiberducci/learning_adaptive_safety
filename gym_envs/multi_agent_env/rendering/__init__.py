from __future__ import annotations
import pathlib
from typing import List, Tuple, Any

from gym_envs.multi_agent_env.common.track import Track
from gym_envs.multi_agent_env.rendering.renderer import EnvRenderer, RenderSpec
from gym_envs.multi_agent_env.rendering.rendering_pygame import PygameEnvRenderer


def make_renderer(
    params: dict[str, Any],
    track: Track,
    agent_ids: list[str],
    render_mode: str = None,
    render_fps: int = 100,
) -> Tuple[EnvRenderer, RenderSpec]:

    cfg_file = pathlib.Path(__file__).parent.absolute() / "rendering.yaml"
    render_spec = RenderSpec.from_yaml(cfg_file)

    if render_mode in ["human", "rgb_array", "human_fast"]:
        renderer = PygameEnvRenderer(
            params=params,
            track=track,
            agent_ids=agent_ids,
            render_spec=render_spec,
            render_mode=render_mode,
            render_fps=render_fps,
        )
    else:
        renderer = None
    return renderer, render_spec
