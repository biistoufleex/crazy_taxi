import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def moving_average(x, periods=5):
    if len(x) < periods:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[periods:] - cumsum[:-periods]) / periods

def bruteforce_taxi(env, num_episodes=10000):
    rewards = []
    steps = []
    success_rate = []
    success_count = 0

    for _ in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0

        while not done:
            action = env.action_space.sample()  # Choose a random action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            episode_steps += 1
            state = next_state

        rewards.append(total_reward)
        steps.append(episode_steps)
        
        if total_reward > 0:
            success_count += 1
        success_rate.append(success_count / (len(rewards)))

    return rewards, steps, success_rate

def plot_rewards_and_steps(rewards, steps):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.5, label='Rewards')
    plt.plot(steps, alpha=0.5, label='Steps')
    plt.plot(moving_average(rewards, 100), color='red', label='Avg Reward')
    plt.plot(moving_average(steps, 100), color='blue', label='Avg Steps')
    plt.title('Rewards and Steps over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('bruteforce_rewards_and_steps.png')
    plt.close()

def plot_success_rate(success_rate):
    plt.figure(figsize=(10, 6))
    plt.plot(success_rate, label='Success Rate')
    plt.title('Success Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    plt.savefig('bruteforce_success_rate.png')
    plt.close()

def plot_reward_distribution(rewards):
    plt.figure(figsize=(10, 6))
    sns.histplot(rewards, kde=True)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.savefig('bruteforce_reward_distribution.png')
    plt.close()

def plot_steps_distribution(steps):
    plt.figure(figsize=(10, 6))
    sns.histplot(steps, kde=True)
    plt.title('Steps Distribution')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.savefig('bruteforce_steps_distribution.png')
    plt.close()

if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    rewards, steps, success_rate = bruteforce_taxi(env)

    plot_rewards_and_steps(rewards, steps)
    plot_success_rate(success_rate)
    plot_reward_distribution(rewards)
    plot_steps_distribution(steps)

    print(f"Average reward over {len(rewards)} episodes: {np.mean(rewards)}")
    print(f"Average steps over {len(steps)} episodes: {np.mean(steps)}")
    print(f"Final success rate: {success_rate[-1]}")