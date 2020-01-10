import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tennis_models import CriticNetwork, ActorNetwork
from dqn_utils import ReplayBuffer, OUNoise, update_model_parameters

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ACTOR_PREFIX = 'actor_'
CRITIC_PREFIX = 'critic_'
AGENT_PREFIX = 'agent_'

class TennisAgent():

    def __init__(self, index, state_size, action_size, num_agents, action_min=-1,
                 action_max=1, buffer_size=int(1e6), learning_frequency=4,
                 training_batch_size=256, gamma=0.99, critic_1st_output=400,
                 critic_2nd_output=300, critic_learning_rate=1e-3,
                 actor_1st_output=400, actor_2nd_output=300,
                 actor_learning_rate=1e-4, tau=1e-3, noise_theta=0.15,
                 noise_sigma=0.2):
        self.index = index
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.action_min = -1
        self.action_max = 1
        self.training_batch_size = training_batch_size
        self.learning_frequency = learning_frequency
        self.gamma = gamma
        self.tau = tau

        self.critic_local_network = self.get_critic_network(first_layer_output=critic_1st_output,
                                                            second_layer_output=critic_2nd_output)
        self.critic_target_network = self.get_critic_network(first_layer_output=critic_1st_output,
                                                             second_layer_output=critic_2nd_output)
        update_model_parameters(tau=1.0, local_network=self.critic_local_network,
                                target_network=self.critic_target_network)
        self.critic_optimizer = optim.Adam(self.critic_local_network.parameters(),
                                           lr=critic_learning_rate)

        self.actor_local_network = self.get_actor_network(first_layer_output=actor_1st_output,
                                                          second_layer_output=actor_2nd_output)
        self.actor_target_network = self.get_actor_network(first_layer_output=actor_1st_output,
                                                           second_layer_output=actor_2nd_output)
        update_model_parameters(tau=1.0, local_network=self.actor_local_network,
                                target_network=self.actor_target_network)
        self.actor_optimizer = optim.Adam(self.actor_local_network.parameters(),
                                          lr=actor_learning_rate)

        self.noise_generator = OUNoise(action_dim=action_size, low=self.action_min,
                                       high=self.action_max,theta=noise_theta,
                                       max_sigma=noise_sigma, min_sigma=noise_sigma)
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                          action_type=np.float32,
                                          training_batch_size=training_batch_size,
                                          device=DEVICE)
        self.step_counter = 0

    def get_critic_network(self, first_layer_output, second_layer_output):
        model = CriticNetwork(state_size=self.state_size,
                              action_size=self.action_size,
                              num_agents=self.num_agents,
                              first_layer_output=first_layer_output,
                              second_layer_output=second_layer_output)

        return model.to(DEVICE)

    def get_actor_network(self, first_layer_output, second_layer_output):
        model = ActorNetwork(state_size=self.state_size, action_size=self.action_size,
                             first_layer_output=first_layer_output,
                             second_layer_output=second_layer_output)

        return model.to(DEVICE)

    def act(self, states, action_parameters):

        state = torch.from_numpy(states[self.index]).float().unsqueeze(0).to(DEVICE)

        self.actor_local_network.eval()
        with torch.no_grad():
            action = self.actor_local_network(state).numpy()
        self.actor_local_network.train()

        add_noise = action_parameters['add_noise']
        if add_noise:
            action = self.noise_generator.get_action(action=action)

        return action


    def step(self, agents, states, actions, rewards, next_states, dones):

        all_states = torch.Tensor(states).view(-1, self.num_agents * self.state_size)
        all_next_states = torch.Tensor(next_states).view(-1, self.num_agents * self.state_size)
        all_actions = torch.Tensor(actions).view(-1, self.num_agents * self.action_size)
        all_rewards = torch.Tensor(rewards).view(-1, self.num_agents)
        all_dones = torch.Tensor(rewards).view(-1, self.num_agents)

        self.replay_buffer.add(state=all_states, action=all_actions, reward=all_rewards,
                               next_state=all_next_states, done=all_dones)

        self.step_counter += 1

        trigger_learning = self.step_counter % self.learning_frequency == 0
        if len(self.replay_buffer) > self.training_batch_size and trigger_learning:

            learning_samples = self.replay_buffer.sample()
            self.learn(agents=agents, learning_samples=learning_samples)

    def learn(self, agents, learning_samples):

        states_sample = learning_samples[0]
        actions_sample = learning_samples[1]
        rewards_sample = learning_samples[2]
        next_states_sample = learning_samples[3]
        dones_sample = learning_samples[4]

        next_actions = self.get_next_actions(agents=agents,
                                             all_next_states=next_states_sample)

        q_values_next_state = self.critic_target_network(next_states_sample,
                                                         next_actions)

        agent_rewards = rewards_sample[:, self.index]
        done_mask = torch.Tensor([np.any(dones_sample[episode].numpy())
                                  for episode in range(len(dones_sample))])

        q_value_current_state = agent_rewards + (self.gamma * q_values_next_state * (1 - done_mask))

        q_value_expected = self.critic_local_network(states_sample, actions_sample)

        critic_loss = F.mse_loss(input=q_value_expected,
                                 target=q_value_current_state)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        agent_states = self.extract_agent_state(all_states=states_sample)
        predicted_actions = self.actor_local_network(agent_states)

        consolidated_actions = actions_sample.clone()
        agent_start = self.index * self.action_size
        agent_end = agent_start + self.action_size
        consolidated_actions[:, agent_start:agent_end] = predicted_actions

        actor_loss = -self.critic_local_network(states_sample, consolidated_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        update_model_parameters(tau=self.tau, local_network=self.critic_local_network,
                                target_network=self.critic_target_network)
        update_model_parameters(tau=self.tau, local_network=self.actor_local_network,
                                target_network=self.actor_target_network)

    def extract_agent_state(self, all_states):

        agent_start = self.index * self.state_size
        agent_end = agent_start + self.state_size
        agent_states = all_states[:, agent_start:agent_end]

        return agent_states

    def get_next_actions(self, agents, all_next_states):

        all_next_actions = torch.FloatTensor()

        for agent in agents:

            agent_next_states = self.extract_agent_state(all_states=all_next_states)
            agent_next_action = agent.actor_target_network(agent_next_states).detach()
            all_next_actions = torch.cat((all_next_actions, agent_next_action), dim=1)

        return all_next_actions

    def reset(self):
        self.noise_generator.reset()

    def save_trained_weights(self, network_file):
        agent_identifier = AGENT_PREFIX + "_" + str(self.index)
        actor_network_file = agent_identifier + "_"  + ACTOR_PREFIX + network_file
        torch.save(self.actor_local_network.state_dict(), actor_network_file)

        critic_network_file = agent_identifier + "_"  + CRITIC_PREFIX + network_file
        torch.save(self.critic_local_network.state_dict(), critic_network_file)

    def load_trained_weights(self, network_file):
        """
        Takes weights from a file and assigns them to the local network.
        """

        agent_identifier = AGENT_PREFIX + "_" + str(self.index)

        actor_network_file = agent_identifier + "_"  + ACTOR_PREFIX + network_file
        self.actor_local_network.load_state_dict(torch.load(actor_network_file))
        print("Actor Network state loaded from ", actor_network_file)

        critic_network_file = agent_identifier + "_"  + CRITIC_PREFIX + network_file
        self.critic_local_network.load_state_dict(torch.load(critic_network_file))
        print("Critic Network state loaded from ", critic_network_file)
