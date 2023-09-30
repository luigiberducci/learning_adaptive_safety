import gymnasium as gym


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip: int):
        self._frame_skip = skip
        super(FrameSkip, self).__init__(env)

    def step(self, action):
        R = 0
        for t in range(self._frame_skip):
            obs, reward, done, truncated, info = self.env.step(action)
            R += reward
            if done or truncated:
                break
        return obs, R, done, truncated, info
