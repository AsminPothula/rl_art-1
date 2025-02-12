import gym
from string_art_env import StringArtEnv

# Initialize the environment
env = StringArtEnv(image_path="target.jpg")

# Reset the environment
state = env.reset()
done = False

# no training loop 

while not done:
    action = env.action_space.sample()  # Randomly select a dest point for the action
    state, reward, done, _ = env.step(action)
    env.render()  # Display the updated canvas

print("Episode finished after 1000 steps.")
