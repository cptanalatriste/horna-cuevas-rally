import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):

    def __init__(self, num_agents, state_size, action_size, first_layer_output,
                 second_layer_output):
        super(CriticNetwork, self).__init__()

        first_layer_input = num_agents * (state_size + action_size)
        self.first_linear_layer = nn.Linear(in_features=first_layer_input,
                                            out_features=first_layer_output)
        self.second_linear_layer = nn.Linear(in_features=first_layer_output,
                                             out_features=second_layer_output)
        self.third_linear_layer = nn.Linear(in_features=second_layer_output,
                                            out_features=1)

    def forward(self, all_states, all_actions):
        data_in_transit = torch.cat((all_states, all_actions), dim=1)

        data_in_transit = F.relu(self.first_linear_layer(data_in_transit))
        data_in_transit = F.relu(self.second_linear_layer(data_in_transit))

        return self.third_linear_layer(data_in_transit)

class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size, first_layer_output,
                 second_layer_output):
        super(ActorNetwork, self).__init__()

        self.first_linear_layer = nn.Linear(in_features=state_size,
                                            out_features=first_layer_output)
        self.second_linear_layer = nn.Linear(in_features=first_layer_output,
                                             out_features=second_layer_output)
        self.third_linear_layer = nn.Linear(in_features=second_layer_output,
                                            out_features=action_size)

    def forward(self, state):
        data_in_transit = F.relu(self.first_linear_layer(state))
        data_in_transit = F.relu(self.second_linear_layer(data_in_transit))

        return F.tanh(self.third_linear_layer(data_in_transit))
