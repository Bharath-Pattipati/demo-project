"""
Reinforcement Learning: Gynasium: https://gymnasium.farama.org/

Gymnasium:
    Space: Space.sample()
    action_space: Env.action_space
    observation_space: Env.observation_space

"""

# %% Import libraries
import gymnasium as gym

# %% Basic Cartpole game
""" env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close() """

# %% Lunar Lander
env = gym.make("LunarLander-v3", render_mode="human")  # Step 1: Create Environment
observation, info = (
    env.reset()
)  # Step 2: Reset Environment and get first observation i.e. initialize environment
print(env.observation_space.shape)

episode_over = False  # Episodic i.e. Finite MDP
while not episode_over:
    action = (
        env.action_space.sample()  # sample random action
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(
        action
    )  # Execute selected action and get reward, return terminated (manually) and truncated (fixed number of steps per episode)
    print(f"Observation: {observation}, \nReward: {reward}")

    episode_over = terminated or truncated

env.close()
