from collections import deque

import numpy as np

class MultiAgentTrainingManager():

    def __init__(self, environment_params):
        self.environment_params = environment_params

    def do_reset(self, environment, agents):
        return []

    def keep_going(self):
        return True

    def get_action_parameters(self):
        return {}

    def do_step(self, environment, actions):
        return ()

    def get_consolidated_score(self, current_scores):
        return current_scores

    def on_episode_end(self, environment, agent, network_file):
        pass

    def start_training(self, agents, environment, score_window, num_episodes,
                       network_file, target_score):
        print("Starting training ...")

        all_scores = []
        avg_scores = []
        last_scores = deque(maxlen=score_window)

        for index, agent in enumerate(agents):
            agent.index = index

        for episode in range(1, num_episodes + 1):
            states = self.do_reset(environment, agents)

            current_scores = np.zeros(len(agents))

            while self.keep_going():
                action_parameters = self.get_action_parameters()

                actions = [agent.act(states=states, action_parameters=action_parameters)
                           for agent in agents]

                next_states, rewards, dones = self.do_step(environment, actions)

                for agent in agents:
                    agent.step(agents=agents, states=states, actions=actions,
                               rewards=rewards, next_states=next_states,
                               dones=dones)

                states = next_states
                current_scores += rewards

                if np.any(dones):
                    break

            consolidated_score = self.get_consolidated_score(current_scores)
            last_scores.append(consolidated_score)
            all_scores.append(consolidated_score)

            average_score = np.mean(last_scores)
            avg_scores.append(average_score)

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, average_score),
                  end="")

            if episode % score_window == 0:
                print("\nEpisode", episode, "Average score over the last", score_window,
                      " episodes: ", average_score)

            if average_score >= target_score:
                print("\nEnvironment solved in ", episode + 1, " episodes. ",
                      "Average score: ", average_score)

                for agent in agents:
                    agent.save_trained_weights(network_file=network_file)
                print("Saving network file at: ", network_file)
                break

            self.on_episode_end(environment, agents, network_file)

        return all_scores, avg_scores
