# horna-cuevas-rally
Two deep-reinforcement learning agents that play tennis, trained using the  [Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm](http://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments).

## Project Details:
Horna and Cuevas are deep-reinforcement learning agents designed for a tailored version of the Tennis environment, from the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).

![The Tennis environment](https://github.com/cptanalatriste/horna-cuevas-rally/blob/master/img/tennis.png?raw=true)

Each agent perceives a state represented via a vector of 24 elements. 
This vector contains information of the position and velocity of the ball and their racket. 
Their actions are composed by vectors of 2 real-valued elements between -1 and 1. These values represent horizontal and vertical movements.

Each agent is rewarded with +0.1 points every time they pass the ball over the net.
If the ball does not cross the net or falls outside the court, the offending agent is penalised with -0.01 points.
We consider the environment solved when **the maximum score among both agents reaches 0.5 on average, over 100 episodes**.

## Getting Started
Before running the agents, be sure to accomplish this first:

1. Clone this repository.
2. Download the Tennis environment appropriate to your operating system (available [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) ).
3. Place the environment file in the cloned repository folder.
4. Setup an appropriate Python environment. Instructions available [here](https://github.com/udacity/deep-reinforcement-learning).

## Instructions
You can start running and training the agents by exploring `Tennis.ipynb`. Also available in the repository:

* `tennis_agent.py` contains the agents' code.
* `tennis_manager.py` has the code for training the agents.
