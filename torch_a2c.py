"""
Pytorch stable baselines 3 A2C model.

This algorithm is the fastest I have tried so far. It can learn at a rate of 
400-500 steps per second. 

At 1,000,000 steps, it converges on a single word and assumes that's always right, 
probably because it only sees a few positive rewards. There are 11M possible actions, 
and to learn english words an algorithm would need to try them all several times, 
so we need to do a few hundred million steps, with very slow learning.

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
