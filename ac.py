"""
Based on:
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

Actor-critic algorithm

Good for continuous action spaces, will it work for our large discrete space?

No luck with this one...
"""

import random
from collections import deque
import gym
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import Add
from keras.optimizers import adam_v2
import keras.backend as K
import tensorflow as tf
from tensorflow import compat
from gym.envs.registration import register
from numpy.random import default_rng

# Cheating - ultimately we don't want the model to know anything about english words
from all_words import WORDS
from wordle import WordleEnv

# Register our custom environment
register(
    id="wordle-v0",
    entry_point="wordle:WordleEnv",
)


class ActorCritic:
    """
    Determines how to assign values to each state, i.e. takes the state
    and action (two-input model) and determines the corresponding value.
    Chain rule: find the gradient of changing the actor network params in
    getting closest to the final value network predictions, i.e. de/dA
    Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #

    """
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.tau = 0.125

        tf.compat.v1.disable_eager_execution()

        self.memory = deque(maxlen=2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = compat.v1.placeholder(
            tf.float32, [None, self.env.action_space.shape[0]]
        )  # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(
            self.actor_model.output, actor_model_weights, -self.actor_critic_grad
        )  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = compat.v1.train.AdamOptimizer(self.learning_rate).apply_gradients(
            grads
        )

        (
            self.critic_state_input,
            self.critic_action_input,
            self.critic_model,
        ) = self.create_critic_model()
        _, _, self.critic_target_model = self.create_critic_model()

        self.critic_grads = tf.gradients(
            self.critic_model.output, self.critic_action_input
        )  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(compat.v1.initialize_all_variables())

    def create_actor_model(self):
        "Create the actor model"

        print("obs shape", self.env.observation_space.shape)
        state_input = Input(shape=self.env.observation_space.shape)
        layer1 = Dense(24, activation="relu")(state_input)
        layer2 = Dense(48, activation="relu")(layer1)
        layer3 = Dense(24, activation="relu")(layer2)
        output = Dense(self.env.action_space.shape[0], activation="relu")(layer3)

        # This model will converge on what looks like 'AAAAA' for all predictions, 
        # but that's beacause it assumes a continuous action space and it's producing 
        # values like 0.11 instead of integers in range 0-25.
        # Need to adapt this to a discrete action space, or find a different algorithm.

        # How do I configure the Model so it understands the actions are [{0-25}, {0-25}, etc] ?
        # The environment defines the action space as
        # spaces.MultiDiscrete([26, 26, 26, 26, 26], dtype=np.ubyte)
        # so I don't understand why actor_model.predict is producing output like
        # [0.         0.12518406 0.         0.         0.05493959]
        # It seems like it's on the right track, since the prios action was
        # NEATH
        # and the prior observation was
        # 02002 <- E and H were in the correct position
        # The fact that positions 1 and 4 are non-zero is encouraging. But why the floats?
        # The observation space shape is (5,) - not specific enough

        model = Model(inputs=state_input, outputs=output)
        adam = adam_v2.Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        "Create the critic model"
        obs_shape = self.env.observation_space.shape
        #print("obs_shape", obs_shape)
        state_input = Input(shape=obs_shape)
        state_h1 = Dense(24, activation="relu")(state_input)
        state_h2 = Dense(48)(state_h1)

        action_shape = self.env.action_space.shape
        #print("action_shape", action_shape)
        action_input = Input(shape=action_shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation="relu")(merged)
        output = Dense(1, activation="relu")(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output)

        adam = adam_v2.Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model


    def remember(self, cur_state, action, reward, new_state, done):
        "Append to memory"
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        "Train the actor"
        for sample in samples:
            cur_state, _, _, _, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(
                self.critic_grads,
                feed_dict={
                    self.critic_state_input: cur_state,
                    self.critic_action_input: predicted_action,
                },
            )[0]

            self.sess.run(
                self.optimize,
                feed_dict={
                    self.actor_state_input: cur_state,
                    self.actor_critic_grad: grads,
                },
            )

    def _train_critic(self, samples):
        "Train the critic"
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.critic_target_model.predict(
                    [new_state, target_action]
                )[0][0]
                reward += self.gamma * future_reward
                #print("_train_critic future reward", reward)

            #print("_train_critic cur_state", cur_state)
            #print("_train_critic action", action)

            self.critic_model.fit([cur_state, action], [reward], verbose=0)

    def train(self):
        "Train"
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    def _update_actor_target(self):
        "Update the target model"
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.critic_target_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.critic_target_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        "Update the critic"
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        "Update the actor and critic models"
        self._update_actor_target()
        self._update_critic_target()


    def act(self, cur_state):
        "Take a random action or exploit what we know"
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            print("act random")
            #return self.env.action_space.sample()

            # CHEATING
            # Let's sample valid words and see if this works
            word = WORDS[default_rng().integers(low=0, high=len(WORDS))]
            return WordleEnv.convert_word(word)
        print("act predict")
        return self.actor_model.predict(cur_state)


def main():
    "Run the training algorithm"
    sess = compat.v1.Session()
    K.set_session(sess)
    env = gym.make("wordle-v0")
    actor_critic = ActorCritic(env, sess)

    for _ in range(100):
        print("New episode")
        cur_state = env.reset()
        action = env.action_space.sample()
        for _ in range(100):
            cur_state = np.reshape(cur_state, (1, env.observation_space.shape[0]))
            action = actor_critic.act(cur_state)
            #print("action", action)
            action = np.reshape(action, (1, env.action_space.shape[0]))
            print("reshaped", action)

            new_state, reward, done, _ = env.step(action[0])
            env.render()
            new_state = np.reshape(new_state, (1, env.observation_space.shape[0]))

            actor_critic.remember(cur_state, action, reward, new_state, done)
            actor_critic.train()

            cur_state = new_state


if __name__ == "__main__":
    main()
