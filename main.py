import gym
import time

env = gym.make('Taxi-v3')


state = env.reset()

env.render()

done = False
while not done:
    # Choose a random action (move) from the available actions
    action = env.action_space.sample()
    
    # Perform the action in the environment and observe the next state, reward, and whether the episode is done
    next_state, reward, done, _ = env.step(action)
    
    # Display the environment screen after taking the action
    env.render()
    
    # Update the current state
    state = next_state

    # wait for 1 second
    time.sleep(1)