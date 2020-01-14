import numpy as np

from rl_manager import MultiAgentTrainingManager

class TennisManager(MultiAgentTrainingManager):

    def __init__(self, environment_params):
        super().__init__(environment_params)

    def do_reset(self, environment, agents):

        brain_name = self.environment_params['brain_name']

        environment_info = environment.reset(train_mode=True)[brain_name]
        return environment_info.vector_observations

    def get_action_parameters(self):
        return {'add_noise': True}

    def do_step(self, environment, actions):
        brain_name = self.environment_params['brain_name']

        environment_info = environment.step(actions)[brain_name]

        next_states = environment_info.vector_observations
        rewards = environment_info.rewards
        dones = environment_info.local_done

        return next_states, rewards, dones

    def get_consolidated_score(self, current_scores):
        return np.max(current_scores)
