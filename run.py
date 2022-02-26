"""
Run the environment with random actions
"""

import gym
from gym.envs.registration import register


register(
    id="wordle-v0",
    entry_point="wordle:WordleEnv",
)

# env = gym.make("CartPole-v1")
# env = gym.make("MountainCar-v0")
# env = gym.make("Taxi-v3")
env = gym.make("wordle-v0")

observation = env.reset()

for _ in range(100000):
    action = env.action_space.sample()
    # print("action", action)
    observation, reward, done, info = env.step(action)
    # print("info", info)

    if reward > 0:
        env.render()

    if done:
        break

env.close()
