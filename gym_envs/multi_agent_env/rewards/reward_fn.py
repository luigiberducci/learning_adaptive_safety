from abc import abstractmethod


class RewardFn:
    def __init__(self, env, **kwargs):
        self.env = env

    @abstractmethod
    def __call__(self, state, action, next_state, done):
        raise NotImplementedError


class DefaultRewardFn(RewardFn):
    def __call__(self, state, action, next_state, done):
        return self.env.timestep
