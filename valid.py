"""
Use the environment to figure out what actions are valid.

Also record which of the valid actions are likely to produce a higher reward.

This file doesn't use any RL algortihms, it's more of a brute force repetition 
using the environment, but without any real "inside knowledge".

This will be useful for any huge action space where only a small percentage of 
actions are valid.
"""

import gym
from numpy.random import default_rng
from gym.envs.registration import register
from wordle import WordleEnv

register(
    id="wordle-v0",
    entry_point="wordle:WordleEnv",
)

NUM_EPISODES = 100
SAMPLES_PER_EPISODE = 1000

class ValidActions():
    "Keep track of which actions resulted in a reward"

    def __init__(self, gymenv, num_episodes, samples_per_episode):
        self.gymenv = gymenv
        self.valid = {}
        self.rewards = {}
        self.num_episodes = num_episodes
        self.samples_per_episode = samples_per_episode
        self.total_samples = 0
        self.dones = 0
        self.samples = None

    def record(self, action, reward):
        "Record a valid action"

        # Convert the action to a key that we can look up and use for sampling later
        key = str(action)
        obj = {
            "action": action, 
            "rewards": [] # Record each time we see it
        }
        if key in self.valid:
            obj = self.valid[key]
        else:
            self.valid[key] = obj
        obj["rewards"].append(reward)

        # Create a lookup by reward so we can rank actions
        if reward not in self.rewards:
            self.rewards[reward] = []
        self.rewards[reward].append(action)

    def learn(self):
        "Step through the action space to learn which actions are valid"
        
        for episode_num in range(self.num_episodes):
            _ = self.gymenv.reset()
            for sample_num in range(self.samples_per_episode):
                action = self.gymenv.action_space.sample()
                # print("action", action)
                _, reward, done, _ = self.gymenv.step(action)
                # print("info", info)
                self.total_samples += 1

                if reward > 0:
                    print(f"Episode {episode_num}, sample {sample_num}")
                    self.gymenv.render()
                    self.record(action, reward)

                if done:
                    self.dones += 1
                    break

    def printout(self):
        "Print out the list of valid actions"
        print(f"{self.num_episodes} episodes, {self.samples_per_episode} samples per episode")
        print(f"{self.total_samples} total samples")
        print(f"{len(self.valid)} valid actions")
        print(f"{len(self.rewards)} different rewards")
        print(f"{self.dones} dones")
        #print("valid", self.valid)
        #print("rewards", self.rewards)
        for reward in self.rewards:
            for action in self.rewards[reward]:
                print(f"{reward}: {WordleEnv.convert_action(action)}")

    def sample(self):
        """
        Choose a sample from the valid actions, favoring actions with higher rewards.
        This is somewhat random, in that it will eventually select all actions in the 
        list, but it weights high rewards so that those actions (common words) are 
        chosen more often. This should be used as the "random" sample in a deep learning 
        algorithm when we are exploring, not exploiting.
        """

        # Create a single array and enter each action into it a number of times according 
        # to its reward. Then take a random selection from the array.
        if self.samples is None:
            self.samples = []
            for reward in self.rewards:
                for _ in range(reward):
                    for action in self.rewards[reward]:
                        self.samples.append(action)

        # Return a sample from the list of valid actions
        return self.samples[default_rng().integers(low=0, high=len(self.samples))]


if __name__ == "__main__":
    envi = gym.make("wordle-v0")
    v = ValidActions(envi, NUM_EPISODES, SAMPLES_PER_EPISODE)
    v.learn()
    v.printout()
    print(f"Sample: {WordleEnv.convert_action(v.sample())}")
    print(f"Sample: {WordleEnv.convert_action(v.sample())}")
    print(f"Sample: {WordleEnv.convert_action(v.sample())}")
    envi.close()
