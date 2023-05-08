import gymnasium as gym


class HabitatEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box()
        self.observation_space = gym.space.Box()

    def step(self, action):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError

    def _gainful_curiosity(self):
        raise NotImplementedError
