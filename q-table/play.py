import numpy as np
import gym
import argparse

def play(env, q_table, episodes):
    print("\nPlaying...")
    for _ in range(episodes):
        state, _ = env.reset()
        state = state if isinstance(state, (int, np.integer)) else state.item()
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state if isinstance(next_state, (int, np.integer)) else next_state.item()
            done = terminated or truncated
            
            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        print(f"Time steps: {epochs}, Penalties: {penalties}")
    env.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the Taxi-v3 game")
    parser.add_argument("-E", "--episodes", type=int, default=3, help="Number of episodes to play")
    args = parser.parse_args()

    env = gym.make("Taxi-v3", render_mode="human")
    q_table = np.load("qtable.npy")
    play(env, q_table, args.episodes)