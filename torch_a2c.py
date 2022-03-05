"""
Pytorch stable baselines 3 A2C model.
"""
import gym

from stable_baselines3 import A2C

from gym.envs.registration import register

# Register our custom environment
register(
    id="wordle-v0",
    entry_point="wordle:WordleEnv",
)

env = gym.make("wordle-v0")

model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.000001)
model.learn(total_timesteps=1000000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
