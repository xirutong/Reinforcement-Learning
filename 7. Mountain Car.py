import gym
import numpy as np
import matplotlib.pyplot as plt
import random

def plot_value_function(env, Q, pos_space, vel_space, episode):
    value_function = np.max(Q, axis=2)
    plt.imshow(value_function, extent=[env.observation_space.low[0], env.observation_space.high[0], 
                                       env.observation_space.low[1], env.observation_space.high[1]], 
               origin='lower', aspect='auto')
    plt.colorbar()
    plt.title(f'Value Function after {episode} episodes')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.show()

def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation, "\nreward: ", reward, "\n")
        if done:
            break

def qlearning(env, alpha=0.1, gamma=0.9, epsilon=1, num_ep=int(15000)):
    # Divide position and velocity into segements
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Between -0.07 and 0.07

    Q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))  # 20×20×3

    epsilon_decay_rate = 2/num_ep
    reaching_goal = 0
    eps_lengths = np.zeros(int(num_ep / 100))
    plot_eps_length = 0

    for episode in range(num_ep):
        state = env.reset()  # the state consists of position and velocity (both starting point being 0)
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        done = False
        render=True if episode%200==0 else False
        rewards = 0
        eps_length = 0

        while (not done) and (eps_length <= 200) and (state[0] <= 0.5):
            eps_length += 1
            plot_eps_length += 1

            # choose action by epsilon greedy policy
            if random.random() < epsilon:  # choose action by epsilon greedy policy
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_p, state_v, :])

            new_state, reward, done, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if render:
                env.render()

            # update Q value using greedy policy
            Q[state_p, state_v, action] = Q[state_p, state_v, action] + alpha * (reward + gamma * np.max(Q[new_state_p, new_state_v, :]) - Q[state_p, state_v, action])

            state =  new_state
            state_p = new_state_p
            state_v = new_state_v

            if state[0] >= env.goal_position:
                reaching_goal += 1

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # plot the value function every 5000 episodes
        """if episode % 5000 == 0 or episode == num_ep-1:
            plot_value_function(env, Q, pos_space, vel_space, episode)"""
        
        # store average episode length over 100 episodes
        if episode % 100 == 0:
            eps_lengths[int(episode / 100)] = plot_eps_length / 100
            plot_eps_length = 0

    return Q, reaching_goal, eps_lengths


def main():
    # repeat the process for 2 times, 10 times takes too long time
    reaching_goal_list = []
    eps_array = np.zeros(int(15000 / 100))

    for i in range(2):
        env = gym.make('MountainCar-v0')
        env.reset()
        print("Running qlearning")
        Q, reaching_goal, eps_lengths = qlearning(env)
        reaching_goal_list.append(reaching_goal)
        print("Repeated whole process for {} time(s) and got number of successes:".format(i+1), reaching_goal)
        eps_array += eps_lengths
        env.close()

    # print the average number of success
    success = np.average(reaching_goal_list)
    print("Average number of success:", success)

    # plot learning curves
    plt.plot(range(15000)[0::100], eps_array / 2)
    plt.xlabel("Episode Number")
    plt.ylabel("Length of episode")
    plt.show()


if __name__ == "__main__":
    main()
