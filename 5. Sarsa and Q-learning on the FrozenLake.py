import random
import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def print_policy(Q, env):
    """ This is a helper function to print a nice policy from the Q function"""
    moves = [u'←', u'↓', u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    policy = np.chararray(dims, unicode=True)
    policy[:] = ' '
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        policy[idx] = moves[np.argmax(Q[s])]
        if env.desc[idx] in [b'H', b'G']:
            policy[idx] = env.desc[idx]
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row]) for row in policy]))


def plot_V(Q, env):
    """ This is a helper function to plot the state values from the Q function"""
    fig = plt.figure()
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    V = np.zeros(dims)
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        V[idx] = np.max(Q[s])
        if env.desc[idx] in ['H', 'G']:
            V[idx] = 0.
    plt.imshow(V, origin='upper',
               extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6,
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y + 0.5, dims[0] - x - 0.5, '{:.3f}'.format(V[x, y]),
                 horizontalalignment='center',
                 verticalalignment='center')
    plt.xticks([])
    plt.yticks([])


def plot_Q(Q, env):
    """ This is a helper function to plot the Q function """
    from matplotlib import colors, patches
    fig = plt.figure()
    ax = fig.gca()

    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape

    up = np.array([[0, 1], [0.5, 0.5], [1, 1]])
    down = np.array([[0, 0], [0.5, 0.5], [1, 0]])
    left = np.array([[0, 0], [0.5, 0.5], [0, 1]])
    right = np.array([[1, 0], [0.5, 0.5], [1, 1]])
    tri = [left, down, right, up]
    pos = [[0.2, 0.5], [0.5, 0.2], [0.8, 0.5], [0.5, 0.8]]

    cmap = plt.cm.RdYlGn
    norm = colors.Normalize(vmin=.0, vmax=.6)

    ax.imshow(np.zeros(dims), origin='upper', extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6, cmap=cmap)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        x, y = idx
        if env.desc[idx] in ['H', 'G']:
            ax.add_patch(patches.Rectangle((y, 3 - x), 1, 1, color=cmap(.0)))
            plt.text(y + 0.5, dims[0] - x - 0.5, '{:.2f}'.format(.0),
                     horizontalalignment='center',
                     verticalalignment='center')
            continue
        for a in range(len(tri)):
            ax.add_patch(patches.Polygon(tri[a] + np.array([y, 3 - x]), color=cmap(Q[s][a])))
            plt.text(y + pos[a][0], dims[0] - 1 - x + pos[a][1], '{:.2f}'.format(Q[s][a]),
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=9, fontweight=('bold' if Q[s][a] == np.max(Q[s]) else 'normal'))

    plt.xticks([])
    plt.yticks([])


def sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.5, num_ep=int(1e6)):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    eps_length = 0
    eps_lengths = np.zeros(int(num_ep / 100))

    # implement the sarsa algorithm
    # This is some starting point performing random walks in the environment:
    for i in range(num_ep):
        s = env.reset()
        done = False

        # choose action by epsilon greedy policy
        if random.random() < epsilon:
            a = random.choice(range(env.action_space.n))
        else:
            a = np.argmax(Q[s])

        while not done:
            eps_length += 1
            s_prime, r, done, _ = env.step(a)  # take one step in the environment

            # choose a' from s' using epsilon greedy policy
            if random.random() < epsilon:
                a_prime = random.choice(range(env.action_space.n))  # choose random action
            else:
                a_prime = np.argmax(Q[s_prime])

            Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[s_prime][a_prime] - Q[s][a])
            s = s_prime
            a = a_prime

        if i % 100 == 0:
            eps_lengths[int(i / 100)] = eps_length / 100  # store average episode length over 100 episodes
            eps_length = 0

    plt.plot(range(num_ep)[0::100], eps_lengths)
    plt.xlabel("Episode Number")
    plt.ylabel("Length of episode")

    return Q


def qlearning(env, alpha=0.1, gamma=0.9, epsilon=0.5, num_ep=int(1e6)):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # implement the qlearning algorithm
    for i in range(num_ep):
        s = env.reset()
        done = False
        # choose action using epsilon-greedy policy (behavior policy)
        while not done:
            # choose action using epsilon-greedy policy (behavior policy)
            if random.random() < epsilon:
                a = random.choice(range(env.action_space.n))
            else:
                a = np.argmax(Q[s])
            s_prime, r, done, _ = env.step(a)
            # update Q value using greedy policy
            Q[s][a] = Q[s][a] + alpha * (r + gamma * np.amax(Q[s_prime]) - Q[s][a])
            s = s_prime
    return Q

#env = gym.make('FrozenLake-v0')
env=gym.make('FrozenLake-v0', is_slippery=False)
#env = gym.make('FrozenLake-v0', map_name="8x8")

print("current environment: ")
env.render()
print()

print("Running sarsa...")
Q = sarsa(env)
plot_V(Q, env)
plot_Q(Q, env)
print_policy(Q, env)
plt.show()

print("\nRunning qlearning")
Q = qlearning(env)
plot_V(Q, env)
plot_Q(Q, env)
print_policy(Q, env)
plt.show()
