import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import numpy as np
import random

class Node:

  def __init__(self, env, parent = None):
    self.state = env
    self.parent = parent
    self.children = []
    self.untried_actions = [action for action in range(action_num)]
    self.visiting_times = 0
    self.q = 0
    self.is_done = False
    self.observation = None
    self.reward = 0
    self.action = None

  def is_fully_expanded(self):
    return len(self.untried_actions) == 0

  def is_terminal_node(self):
    return self.is_done

  def compute_mean_value(self):
    if self.visiting_times == 0:
      return 0
    return self.q / self.visiting_times

  def best_child(self):
    scores = [child.compute_mean_value() for child in self.children]
    child_index = np.argmax(scores)
    return self.children[child_index]

  def expand(self):
    action = self.untried_actions.pop()
    next_state = copy(self.state)
    self.observation, self.reward, self.is_done,_ = next_state.step(action)
    child_node = Node(next_state, parent = self)
    child_node.action = action
    self.children.append(child_node)
    return child_node
  
  def rollout(self, t_max = 100):
    """ Random policy for rollouts """
    state = copy(self.state)
    rollout_return = 0
    done = False
    for _ in range(t_max):
      action = env.action_space.sample()
      _, reward, done, _ = state.step(action)
      rollout_return += reward
      if done:
        break
    return rollout_return

  def backpropagate(self, child_value):
    node_value = self.reward + child_value
    self.q += node_value
    self.visiting_times += 1
    if self.parent:
      return self.parent.backpropagate(node_value)


class MonteCarloTreeSearch(object):
  def __init__(self, node):
    self.root = node

  def best_action(self, simulations_number=500):
    for _ in range(0, simulations_number):
      v = self._tree_policy()
      reward = v.rollout()
      v.backpropagate(reward)
    return self.root.best_child()

  def _tree_policy(self, epsilon = 0.1):
    current_node = self.root
    while not current_node.is_terminal_node():
      if not current_node.is_fully_expanded():
        return current_node.expand()
      else:
        if random.random() < epsilon:
          current_node = random.choice(current_node.children)
        else:
          current_node = current_node.best_child()
    return current_node

env = gym.make('Taxi-v3').env
action_num = env.action_space.n
state_num = env.observation_space.n
rewards = []

for i in range(10):
  env.reset()
  env.render()

  root = Node(env)
  is_done = False
  total_reward, epochs = 0, 0

  while not is_done:
      env.render()
      mcts = MonteCarloTreeSearch(root)
      best_child = mcts.best_action()
      new_state, reward, is_done, info = env.step(best_child.action)
      total_reward += reward
      epochs += 1
      root = best_child

  env.render()
  print('Timesteps taken:', epochs)
  print("finished run " + str(i+1) + " with reward: " + str(total_reward))
  rewards.append(total_reward)
  
print("mean reward: ", np.mean(rewards))