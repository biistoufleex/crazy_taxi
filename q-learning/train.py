import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import argparse
import time
import gymnasium as gym
import seaborn as sns

def moving_average(x: list, periods: int = 5) -> list:
    if len(x) < periods:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    res = (cumsum[periods:] - cumsum[:-periods]) / periods
    return np.hstack([x[:periods - 1], res])

def plot_durations(episode_durations: list,
                   reward_in_episode: list,
                   epsilon_vec: list,
                   max_steps_per_episode: int = 100) -> None:
    lines = []
    fig = plt.figure(1, figsize=(15, 7))
    plt.clf()
    ax1 = fig.add_subplot(111)

    plt.title(f'Training...')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Duration & Rewards')
    ax1.set_ylim(-2 * max_steps_per_episode, max_steps_per_episode + 10)
    ax1.plot(episode_durations, color="C1", alpha=0.2)
    ax1.plot(reward_in_episode, color="C2", alpha=0.2)
    mean_steps = moving_average(episode_durations, periods=5)
    mean_reward = moving_average(reward_in_episode, periods=5)
    lines.append(ax1.plot(mean_steps, label="steps", color="C1")[0])
    lines.append(ax1.plot(mean_reward, label="rewards", color="C2")[0])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon')
    lines.append(ax2.plot(epsilon_vec, label="epsilon", color="C3")[0])
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc=3)

    plt.savefig("QLearning_graph.png")
    plt.close()

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
    plt.savefig('QLearning_rewards_and_steps.png')
    plt.close()

def plot_success_rate(success_rate):
    plt.figure(figsize=(10, 6))
    plt.plot(success_rate, label='Success Rate')
    plt.title('Success Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    plt.savefig('QLearning_success_rate.png')
    plt.close()

def plot_reward_distribution(rewards):
    plt.figure(figsize=(10, 6))
    sns.histplot(rewards, kde=True)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.savefig('QLearning_reward_distribution.png')
    plt.close()

def plot_steps_distribution(steps):
    plt.figure(figsize=(10, 6))
    sns.histplot(steps, kde=True)
    plt.title('Steps Distribution')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.savefig('QLearning_steps_distribution.png')
    plt.close()

def train(env=gym.make("Taxi-v3"),
          episodes: int = 25000,
          lr: float = 0.1,
          gamma: float = 0.99,
          epsilon: float = 0.3,
          max_epsilon: float = 1,
          min_epsilon: float = 0.001,
          epsilon_decay: float = 0.1,
          path_table: str = "qtable") -> tuple[float, int]:
    
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    start_date = datetime.now()
    start_time = time.time()
    total_reward = []
    steps_per_episode = []
    epsilon_vec = []
    success_rate = []
    success_count = 0

    print("{} - Starting Training...\n".format(start_date))
    start_episode = time.time()
    for e in range(episodes):
        state, _ = env.reset()
        state = state.item() if isinstance(state, np.ndarray) else state

        done = False
        episode_reward = 0
        episode_steps = 0
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * e)
        epsilon_vec.append(epsilon)

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_state.item() if isinstance(next_state, np.ndarray) else next_state
            episode_reward += reward
            episode_steps += 1

            current_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            q_table[state, action] = (1 - lr) * current_value + lr * (reward + gamma * next_max)
            state = next_state

        total_reward.append(episode_reward)
        steps_per_episode.append(episode_steps)

        if episode_reward > 0:
            success_count += 1
        success_rate.append(success_count / (e + 1))

        if e % int(episodes / 100) == 0:
            episode_time = (time.time() - start_episode)
            print(
                "[EPISODE {}/{}] - Mean reward for last {} Episodes: {} in {} steps - Mean Time Per Episode: {}"
                .format(e, episodes, int(episodes / 100),
                        np.mean(total_reward[-int(episodes / 100):]),
                        np.mean(steps_per_episode[-int(episodes / 100):]),
                        np.round(episode_time / (e + 1), 6)))

    plot_durations(steps_per_episode, total_reward, epsilon_vec, max_steps_per_episode=200)
    plot_rewards_and_steps(total_reward, steps_per_episode)
    plot_success_rate(success_rate)
    plot_reward_distribution(total_reward)
    plot_steps_distribution(steps_per_episode)

    end_date = datetime.now()
    execution_time = (time.time() - start_time)

    print()
    print("{} - Training Ended".format(end_date))
    print("Mean Reward: {}".format(np.mean(total_reward)))
    print("Time to train: \n    - {}s\n    - {}min\n    - {}h".format(
        np.round(execution_time, 2), np.round(execution_time / 60, 2),
        np.round(execution_time / 3600, 2)))
    print("Mean Time Per Episode: {}".format(
        np.round(execution_time / len(total_reward), 6)))

    np.save(path_table, q_table)

    return np.round(execution_time, 2), np.mean(total_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Taxi Driver Using the Q-Learning Algorithm")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100000,
        help="Number of episodes",
    )
    parser.add_argument("-l",
                        "--learning_rate",
                        type=float,
                        default=0.01,
                        help="Learning Rate")
    parser.add_argument("-g",
                        "--gamma",
                        type=float,
                        default=0.99,
                        help="Discount Rating")
    parser.add_argument("-e",
                        "--epsilon",
                        type=float,
                        default=1,
                        help="Exploration Rate")
    parser.add_argument("--min_epsilon",
                        type=float,
                        default=0.001,
                        help="Minimal value for Exploration Rate")
    parser.add_argument("-d",
                        "--decay_rate",
                        type=float,
                        default=0.01,
                        help="Exponential decay rate for Exploration Rate")

    args = parser.parse_args()

    epsilon = args.epsilon
    max_epsilon = args.epsilon
    episodes = args.episodes
    lr = args.learning_rate
    gamma = args.gamma
    min_epsilon = args.min_epsilon
    epsilon_decay = args.decay_rate

    env = gym.make("Taxi-v3")

    execution_time, mean_reward = train(env, episodes, lr, gamma, epsilon, max_epsilon,
                         min_epsilon, epsilon_decay)
    
    print(f"Execution time: {execution_time} seconds")
    print(f"Mean reward: {mean_reward}")