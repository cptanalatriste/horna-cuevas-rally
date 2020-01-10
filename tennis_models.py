import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def get_fan_in_limit(layer):
    fan_in = layer.weight.data.size()[0]
    limit = 1. / np.sqrt(fan_in)
    return limit

class CriticNetwork(nn.Module):

    def __init__(self, num_agents, state_size, action_size, first_layer_output,
                 second_layer_output):
        super(CriticNetwork, self).__init__()

        first_layer_input = num_agents * state_size
        self.first_linear_layer = nn.Linear(in_features=first_layer_input,
                                            out_features=first_layer_output)

        second_layer_input = first_layer_output + num_agents * action_size
        self.second_linear_layer = nn.Linear(in_features=second_layer_input,
                                             out_features=second_layer_output)
        self.third_linear_layer = nn.Linear(in_features=second_layer_output,
                                            out_features=1)

        self.init_weights()

    def forward(self, all_states, all_actions):
        data_in_transit = F.relu(self.first_linear_layer(all_states))
        data_in_transit = torch.cat((data_in_transit, all_actions), dim=1)
        data_in_transit = F.relu(self.second_linear_layer(data_in_transit))

        return self.third_linear_layer(data_in_transit)

    def init_weights(self):

        first_layer_limit = get_fan_in_limit(self.first_linear_layer)
        self.first_linear_layer.weight.data.uniform_(-first_layer_limit,
                                                     first_layer_limit)

        second_layer_limit = get_fan_in_limit(self.second_linear_layer)
        self.second_linear_layer.weight.data.uniform_(-second_layer_limit,
                                                      second_layer_limit)

        third_layer_limit = 3e-3
        self.third_linear_layer.weight.data.uniform_(-third_layer_limit,
                                                     third_layer_limit)



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

        self.init_weights()

    def forward(self, state):
        data_in_transit = F.relu(self.first_linear_layer(state))
        data_in_transit = F.relu(self.second_linear_layer(data_in_transit))

        return F.tanh(self.third_linear_layer(data_in_transit))

    def init_weights(self):

        first_layer_limit = get_fan_in_limit(self.first_linear_layer)
        self.first_linear_layer.weight.data.uniform_(-first_layer_limit,
                                                     first_layer_limit)

        second_layer_limit = get_fan_in_limit(self.second_linear_layer)
        self.second_linear_layer.weight.data.uniform_(-second_layer_limit,
                                                      second_layer_limit)

        third_layer_limit = 3e-3
        self.third_linear_layer.weight.data.uniform_(-third_layer_limit,
                                                     third_layer_limit)
