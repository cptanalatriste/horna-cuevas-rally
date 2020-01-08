import numpy as np

class TennisAgent():

    def __init__(self, state_size, action_size, action_min=-1, action_max=1):
        self.state_size = state_size
        self.action_size = action_size
        self.action_min = -1
        self.action_max = 1

    def act(self, state, action_parameters):
        action = np.random.randn(self.action_size)
        action = np.clip(action, a_min=self.action_min, a_max=self.action_max)

        return action
