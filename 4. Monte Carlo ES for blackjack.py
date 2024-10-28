import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')


def single_run_20():
    """ run the policy that sticks for >= 20 """
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    ret = 0.
    while not done:
        print("observation:", obs)
        states.append(obs)

        if obs[0] >= 20:
            print("stick")
            obs, reward, done, _ = env.step(0)  # step=0 for stick
        else:
            print("hit")
            obs, reward, done, _ = env.step(1)  # step=1 for hit
        print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    print("final observation:", obs)
    print(f"this is the state: {states}")
    return states, ret


def policy_evaluation():
    np.random.seed(2024)
    """ Implementation of first-visit Monte Carlo prediction """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))
    visits = np.zeros((10, 10, 2))
    maxiter = 500000  # use whatever number of iterations you want

    for i in range(maxiter):
        states, ret = single_run_20()
        visited_state = set()

        for state in reversed(states):
            usable_index = int(state[2])
            player_index = state[0] - 12
            dealer_index = state[1] - 1
            index_whole = (player_index, dealer_index, usable_index)

            if index_whole not in visited_state:
                visits[index_whole] += 1
                returns[index_whole] += ret

            visited_state.add(index_whole)

            if visits[index_whole] != 0:
                V[index_whole] = returns[index_whole] / visits[index_whole]

    fig, axes = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'}, figsize=(16, 8))

    x = np.arange(1, 11)  # Dealer's card: 1-10
    y = np.arange(12, 22)  # Player's sum: 12-21
    X, Y = np.meshgrid(x, y)

    # Plot for usable ace
    axes[0].plot_surface(X, Y, V[:, :, 1], cmap='viridis', edgecolor='none')
    axes[0].set_title('Usable Ace')
    axes[0].set_xlabel('Dealer Showing')
    axes[0].set_ylabel('Player Sum')
    axes[0].set_zlabel('Value')

    # Plot for no usable ace
    axes[1].plot_surface(X, Y, V[:, :, 0], cmap='viridis', edgecolor='none')
    axes[1].set_title('No Usable Ace')
    axes[1].set_xlabel('Dealer Showing')
    axes[1].set_ylabel('Player Sum')
    axes[1].set_zlabel('Value')

    fig.suptitle(f"Value Function After {maxiter} Episodes")
    plt.show()

    return V


def monte_carlo_es():
    """ Implementation of Monte Carlo ES """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    pi = np.zeros((10, 10, 2))
    Q = np.ones((10, 10, 2, 2)) * 100  # recommended: optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))
    maxiter = 20000000  # use whatever number of iterations you want

    for i in range(maxiter):
        action_index = np.random.randint(0, 2)
        obs = env.reset()
        done = False
        states = []
        ret = 0.
        visited_state = set()
        start = True

        while not done:
            if start:
                states.append((obs, action_index))
                obs, reward, done, _ = env.step(action_index)
                start = False
            else:
                pi_action = (obs[0] - 12, obs[1] - 1, int(obs[2]))
                states.append((obs, int(pi[pi_action])))
                obs, reward, done, _ = env.step(int(pi[pi_action]))
            ret += reward
            # Note that gamma = 1. in this exercise

        for state, action in reversed(states):
            usable_index = int(state[2])
            player_index = state[0] - 12
            dealer_index = state[1] - 1
            state_index = (player_index, dealer_index, usable_index)
            index_whole = (player_index, dealer_index, usable_index, action)

            if index_whole not in visited_state:
                visits[index_whole] += 1
                returns[index_whole] += ret
            visited_state.add(index_whole)

            if visits[index_whole] != 0:
                Q[index_whole] = returns[index_whole] / visits[index_whole]
                pi[state_index] = Q[state_index].argmax()

        if i % 100000 == 0:
            print("Iteration: " + str(i))
            print(pi[:, :, 0])
            print(pi[:, :, 1])


def main():
    single_run_20()
    policy_evaluation()
    monte_carlo_es()


if __name__ == "__main__":
    main()
