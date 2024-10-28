import numpy as np
import matplotlib.pyplot as plt
import random


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # init variables (rewards, n_plays, Q) by playing each arm once
    for a in possible_arms:
        rewards[a] = bandit.play_arm(a)
        n_plays[a] += 1
        Q[a] = rewards[a]

    # Main loop
    while bandit.total_played < timesteps:
        a = np.argmax(Q)
        reward_for_a = bandit.play_arm(a)
        rewards[a] += reward_for_a
        n_plays[a] += 1
        Q[a] = rewards[a] / n_plays[a]


def epsilon_greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)
    epsilon = 0.1

    reward_for_a = bandit.play_arm(0)  # Just play arm 0 as placeholder
    rewards[0] = reward_for_a
    n_plays[0] += 1
    Q[0] = reward_for_a

    while bandit.total_played < timesteps:
        if random.random() < epsilon:  # Exploration (epsilon case)
            a = random.choice(possible_arms)
            reward_for_a = bandit.play_arm(a)

        else:  # Exploitation (greedy case)
            a = np.argmax(Q)
            reward_for_a = bandit.play_arm(a)

        rewards[a] += reward_for_a
        n_plays[a] += 1
        Q[a] = rewards[a] / n_plays[a]


def main():
    n_episodes = 10000  # set to 10000 to decrease noise in plot
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    main()
