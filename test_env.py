import pathlib

import matplotlib.pyplot as plt
import yaml

from training_env_factory import make_env_factory, get_env_id


def main():
    env_id = "f110-multi-agent-v1"
    use_cbf = True
    use_ctrl = True

    env_id = get_env_id(env_id, use_cbf, use_ctrl)
    cfg_path = pathlib.Path(__file__).parent / "gym_envs" / "cfgs" / f"{env_id}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Env config file {cfg_path} does not exist.")

    with open(cfg_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        base_env_id = data["base_env"]
        env_params = data["env_params"]
        cbf_params = data["cbf_params"]

    make_env = make_env_factory(env_id=base_env_id)
    env = make_env(env_id=base_env_id, idx=0, evaluation=False,
                   env_params=env_params, cbf_params=cbf_params,
                   capture_video=False, log_dir=None,
                   default_render_mode="human")()

    done = False
    env.reset()
    frames = []
    while not done:
        action = env.action_space.sample()
        obs, reward, trunc, term, info = env.step(action)
        done = trunc or term
        last_frames = env.render()
        if last_frames is not None:
            frames.append(last_frames)

    env.close()

    # animate
    if len(frames) > 0:
        import imageio
        imageio.mimsave('movie.gif', frames)


if __name__=="__main__":
    main()