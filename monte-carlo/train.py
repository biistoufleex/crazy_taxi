import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import argparse
import random
import time
import gymnasium as gym
import progressbar

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
    '''Plot graphs containing Epsilon, Rewards, and Steps per episode over time'''
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

    return


def plot_durations(episode_durations: list,
                   reward_in_episode: list,
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

    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc=3)

    return

def train(env=gym.make("Taxi-v3"),
                      episodes: int = 25000,
                      gamma: float = 0.99,
                      path_table: str = "monte_carlo_table",
                      path_graph: str = "MonteCarlo_graph.png") -> tuple[float, int]:
    
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    returns = {(s, a): [] for s in range(env.observation_space.n) for a in range(env.action_space.n)}

    start_date = datetime.now()
    start_time = time.time()
    total_reward = []
    steps_per_episode = []

    print("{} - Starting Training...\n".format(start_date))
    start_episode = time.time()
    bar = progressbar.ProgressBar()
    bar(range(episodes))
    bar.start()
    
    for e in range(episodes):
        state, _ = env.reset()
        state = state.item() if isinstance(state, np.ndarray) else state

        episode = []
        done = False
        total_reward.append(0)
        steps_per_episode.append(0)
        display_episode = random.uniform(0, 1) < 0.001

        while not done:
            action = np.argmax(q_table[state]) if random.uniform(0, 1) > 0.1 else env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state.item() if isinstance(next_state, np.ndarray) else next_state
            
            episode.append((state, action, reward))
            total_reward[e] += reward
            steps_per_episode[e] += 1
            state = next_state

            if display_episode:
                env.render()

        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if (state, action) not in [(x[0], x[1]) for x in episode[0:t]]:
                returns[(state, action)].append(G)
                q_table[state, action] = np.mean(returns[(state, action)])

        if e % int(episodes / 100) == 0:
            episode_time = (time.time() - start_episode)
            print(
                "[EPISODE {}/{}] - Mean reward for last {} Episodes: {} in {} steps - Mean Time Per Episode: {}"
                .format(e, episodes, int(episodes / 100),
                        np.mean(total_reward[-int(episodes / 100):]),
                        np.mean(steps_per_episode[-int(episodes / 100):]),
                        np.round(episode_time / e, 6) if e != 0 else 0))
            
        bar.update(e)
    
    bar.finish()

    plot_durations(steps_per_episode,
                   total_reward,
                   max_steps_per_episode=200)
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
    plt.savefig(path_graph)

    return np.round(execution_time, 2), np.mean(total_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Taxi Driver Using the Monte Carlo Algorithm")
    parser.add_argument(
        "-E",
        "--episodes",
        type=int,
        default=25000,
        help="Number of episodes",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.99,
        help="Discount Rating"
    )

    args = parser.parse_args()

    episodes = args.episodes
    gamma = args.gamma

    env = gym.make("Taxi-v3")

    time, reward = train(env, episodes, gamma)