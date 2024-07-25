import numpy as np
import gymnasium as gym
from tqdm import tqdm
import argparse

def bruteforce_taxi(env, num_episodes=10000):
    print("\nPlaying...")
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        state = state if isinstance(state, (int, np.integer)) else state.item()
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            action = env.action_space.sample()  # Choose a random action
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state if isinstance(next_state, (int, np.integer)) else next_state.item()
            done = terminated or truncated
            
            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        print(f"Episode {episode + 1}: Time steps: {epochs}, Penalties: {penalties}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the Taxi-v3 game using bruteforce")
    parser.add_argument("-E", "--episodes", type=int, default=3, help="Number of episodes to play")
    args = parser.parse_args()

    env = gym.make("Taxi-v3", render_mode="human")
    bruteforce_taxi(env, args.episodes)