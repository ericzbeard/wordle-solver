"""
Based on:
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

Not sure if this is the right algorithm... action space is too big.

I think this won't work for a large, MultiDiscrete action space.

https://www.reddit.com/r/reinforcementlearning/comments/hp95c6/multi_discrete_action_spaces_for_dqn/

"""

import random
from collections import deque
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2

from gym.envs.registration import register

# Register our custom environment
register(
    id="wordle-v0",
    entry_point="wordle:WordleEnv",
)


class DQN:
    "Deep Q Network"

    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        # Hyperparameters

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = 0.125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        "Create the neural network using Keras"
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))

        # I think this is why maybe DQN is not the right algorithm
        # The action space is huge.
        # model.add(Dense(self.env.action_size()))
        model.add(Dense(5)) # Is this the entire action space or the length of the shape?

        opt = adam_v2.Adam(learning_rate=self.learning_rate)  # , decay=lr/epochs

        model.compile(loss="mean_squared_error", optimizer=opt)
        return model

    def act(self, state):
        "Decide if we're going to do something random, or exploit what we know"
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # Breaks here. Probably because we aren't reshaping in main().
        #
        # ValueError: Exception encountered when calling layer "sequential" (type Sequential).
        # Input 0 of layer "dense" is incompatible with the layer:
        # expected min_ndim=2, found ndim=1.
        # Full shape received: (None,)
        # Call arguments received:
        #   • inputs=tf.Tensor(shape=(None,), dtype=int64)
        #   • training=False
        #   • mask=None

        # After fixing reshape, action retuned is an int64, which fails

        print("act state", state)
        prediction = self.model.predict(state)
        print("prediction", prediction)
        return prediction[0]

        # return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        "Store the results of each step"
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        "Learn from what we saw in the past"
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            print("sample", sample)
            # pylint:disable=unpacking-non-sequence
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            print("target", target)
            if done:
                target[0][action] = reward
            else:
                q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        "Copy weights from the main model to the target"
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (
                1 - self.tau
            )
        self.target_model.set_weights(target_weights)

    def save_model(self, filename):
        "Save the model"
        self.model.save(filename)


def main():
    "Train the model"

    env = gym.make("wordle-v0")

    trials = 1
    trial_len = 500

    dqn_agent = DQN(env=env)
    for trial in range(trials):
        print("Starting trial", trial)
        # cur_state = np.reshape(env.reset(), (1,2))
        # cur_state = env.reset().reshape(1, 2)
        cur_state = np.reshape(env.reset(), (1, 5))
        print("cur_state", cur_state)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            print("action", action)
            new_state, reward, done, _ = env.step(action)

            # if reward > 0:
            env.render()

            # new_state = np.reshape(new_state, (1, 2))
            # new_state = new_state.reshape(1, 2)
            new_state = np.reshape(new_state, (1, 5))
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break
        if step >= 199:
            print(f"Failed to complete in trial {trial}")
            if step % 10 == 0:
                dqn_agent.save_model(f"trial-{trial}.model")
        else:
            print(f"Completed in {trial} trials")
            dqn_agent.save_model("success.model")
            break


if __name__ == "__main__":
    main()
