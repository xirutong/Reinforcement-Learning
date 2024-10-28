import gym
import numpy as np
import matplotlib.pyplot as plt
import random

def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e3)):
    """ TODO: implement the n-step sarsa algorithm """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode_length_sum = 0
    
    for _ in range(num_ep):
        s = env.reset()
        done = False

        # choose action by epsilon greedy policy
        if random.random() < epsilon:
            a = random.choice(range(env.action_space.n))
        else:
            a = np.argmax(Q[s])

        # initialize state, action, reward and time step lists
        states = [s]
        actions = [a]
        rewards = [0]

        t = 0
        T = float('inf')
        tau = 0


        while tau < (T - 1):
            episode_length_sum += 1
            if t < T:
                s_prime, r, done, _ = env.step(a)  # take one step in the environment
                states.append(s_prime)
                rewards.append(r)

                if done:
                    T = t + 1
                else:
                    # choose a' from s' using epsilon greedy policy
                    if random.random() < epsilon:
                        a_prime = random.choice(range(env.action_space.n))
                    else:
                        a_prime = np.argmax(Q[s_prime])
                    
                    actions.append(a_prime)
            
            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1)])

                if tau + n < T:
                     G += (gamma ** n) * Q[states[tau + n]][actions[tau + n]]

                Q[states[tau]][actions[tau]] += alpha * (G - Q[states[tau]][actions[tau]])

            t += 1
          
    average_length = episode_length_sum/num_ep

    return Q,  average_length


# use value iteration to get real Q values
def value_iteration(env, gamma=0.8, theta=1e-10):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V_states = np.zeros(n_states)

    iteration = 0
    while True:
        delta = 0
        for state in range(n_states):
            v = V_states[state]
            max_value = float('-inf')
            for action, tuples in env.P[state].items():
                value_each_action = sum(p * (r + gamma * V_states[n_state]) for p, n_state, r, _ in tuples)
                max_value = max(max_value, value_each_action)
            V_states[state] = max_value
            delta = max(delta, abs(v - max_value))
        
        iteration += 1

        if delta < theta:
            #print(f"Number of iterations: {iteration}")
            #print("Optimal value function:")
            #print(V_states)
            break

    # Compute the Q-values based on the optimal value function V
    Q_table = np.zeros((n_states, n_actions))
    for state in range(n_states):
        for action, tuples in env.P[state].items():
            Q_table[state, action] = sum(p * (r + gamma * V_states[n_state]) for p, n_state, r, _ in tuples)

    # Derive the policy from the Q-values
    policy = np.argmax(Q_table, axis=1)

    return Q_table


# Running experiments
env = gym.make('FrozenLake-v0', map_name="8x8")
n_values = [1, 2, 4, 8, 16, 32]
alpha_values = np.linspace(0.01, 1.0, 20)
num_trials = 20

# plot average RMS error
def evaluate_rms_error(nstepQ, realQ):
    assert nstepQ.shape == realQ.shape, "Q-tables must have the same shape"
    rms_error = np.sqrt(np.mean((nstepQ - realQ) ** 2))

    return rms_error

realQ = value_iteration(env=env)

avg_rms = {n:[] for n in n_values}
for n in n_values:
    for alpha in alpha_values:
            rms_errors=[]
            for _ in range(num_trials):
                    Q_n_step, _ = nstep_sarsa(env, n=n, alpha=alpha)
                    rms_errors.append(evaluate_rms_error(nstepQ=Q_n_step, realQ=realQ))
            average_rms = np.mean(rms_errors)
            avg_rms[n].append(average_rms)
            print(f'n = {n}, alpha = {alpha:.2f}, average rms = {average_rms}')

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 8))

for n in n_values:
    plt.plot(alpha_values, avg_rms[n], label=f"n={n}")

ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('average RMS Error')
ax.set_title(r'Average RMS Error over Different Values of $n$ and $\alpha$')
ax.legend()
ax.grid(True)

plt.show()


# plot average episode length
"""
avg_length = {n:[] for n in n_values}

for n in n_values:
    for alpha in alpha_values:
            lengths=[]
            for _ in range(num_trials):
                    Q_n_step, average_episode_length = nstep_sarsa(env, n=n, alpha=alpha)
                    lengths.append(average_episode_length)
            average_length = np.mean(lengths)
            avg_length[n].append(average_length)
            print(f'n = {n}, alpha = {alpha:.2f}, average length = {average_length}')

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 8))

for n in n_values:
    plt.plot(alpha_values, avg_length[n], label=f"n={n}")

ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Steps per episode')
ax.legend()
ax.grid(True)

plt.show()"""