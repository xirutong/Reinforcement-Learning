import gym
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def policy(state, theta):
    """ return probabilities for actions under softmax action selection """
    z = np.dot(state, theta)
    return softmax(z)

def generate_episode(env, theta, display=False):
    """ generates one episode and returns the list of states,
        the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def discount_rewards(rewards, gamma=0.9):
    """Compute discounted rewards."""
    discounted = np.zeros_like(rewards)
    cumulative = 0
    for i in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[i]
        discounted[i] = cumulative
    return discounted


def REINFORCE(env, gamma=0.99, learning_rate=0.001, num_ep=10000, baseline=False):
    theta = np.random.rand(4, 2)  # policy parameters
    w = np.random.rand(4)
    episode_lengths = []
    mean_episode_lengths = []

    for e in range(num_ep):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        episode_lengths.append(len(states))
        discounted_rewards = discount_rewards(rewards, gamma)

        # Convert lists to numpy arrays for easier manipulation
        states = np.array(states)
        actions = np.array(actions)
        discounted_rewards = np.array(discounted_rewards)

        # Compute value function estimates
        values = np.dot(states, w)
        advantages = discounted_rewards - values

        for t in range(len(states)):
            w += learning_rate * advantages[t] * states[t]

        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            Gt = discounted_rewards[t]
            probs = policy(state, theta)

            grad = np.zeros_like(theta)
            for a in range(env.action_space.n):
                if a == action:
                    grad[:, a] = state * (1 - probs[a])
                else:
                    grad[:, a] = -state * probs[a]
            if baseline:
                theta += learning_rate * advantages[t] * grad
            else:
                theta += learning_rate * Gt * grad

        # Track mean episode length of the last 100 episodes
        if len(episode_lengths) >= 100:
            mean_episode_lengths.append(np.mean(episode_lengths[-100:]))
        else:
            mean_episode_lengths.append(np.mean(episode_lengths))

        if e % 100 == 0:
            print(f"Episode {e}, Mean of last 100 episode lengths: {mean_episode_lengths[-1]}")
        if mean_episode_lengths[-1] >= 495:
            print(f"Goal achieved after {e + 1} episodes!")
            break

    return mean_episode_lengths


def main():
    env = gym.make('CartPole-v1')

    print("Running REINFORCE with baseline...")
    mean_episode_lengths = REINFORCE(env, baseline=True)
    env.close()

    # Plotting
    plt.plot(mean_episode_lengths)
    plt.xlabel('Episode Count')
    plt.ylabel('Mean Episode Length (Last 100)')
    plt.title('Mean Episode Length Over Time (with baseline)')
    plt.show()


if __name__ == "__main__":
    main()
