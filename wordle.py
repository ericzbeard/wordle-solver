"""
Wordle Gym Environment
"""

import sys
from contextlib import closing
from io import StringIO
from typing import Optional

import numpy as np
from numpy.random import default_rng
from gym import Env, spaces

from all_words import WORDS

INVALID_WORD = -10
VALID_WORD = 1
MISPLACED_LETTER = 5
CORRECT_LETTER = 10

BLACK = "⬛"
YELLOW = "🟨"
GREEN = "🟩"


class WordleEnv(Env):
    """
    An environment that describes a Wordle game.

    Wordle is a daily puzzle, now hosted on NYT, that is based on 5-letter English words.

    After guessing a Word, which is not accepted unless it's in the list of valid words,
    the player is given hints about what letters are correct and in the right position.

    ⬛: A letter that's not in the word
    🟨: A letter that is in the word but not in the correct location
    🟩: A letter that is in the correct location

    For example, if the word is HELLO and the guess is HOUSE, the clue would be 🟩🟨⬛⬛🟨.

    The observation space can be represented by a 5 element array with black, yellow, green
    being represented by 0,1,2 for simplicity.

    The action space is much bigger, a 5 element array of ASCII decimals, which
    yields 11,881,376 possible combinations.

    Rewards and Penalties

    Invalid English words are penalized, correct words get a small reward, and additional rewards
    increase with the number of correct letter.
    """

    answer = ""

    def __init__(self):

        self.action_space = spaces.MultiDiscrete([26, 26, 26, 26, 26], dtype=np.ubyte)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3, 3], dtype=np.ubyte)
        self.reward_range = [INVALID_WORD, CORRECT_LETTER * 5]

        self.last_step = WordleEnv.empty_step()

        self.reset()

    def set_answer(self, test):
        "Set a specific answer for testing"
        self.reset()
        self.answer = test

    @staticmethod
    def empty_step():
        "Return an empty step object for rendering"
        return ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], 0, False, {})

    def step(self, action):
        "Evaluate a single action"

        observation = [0, 0, 0, 0, 0]
        reward = 0
        done = False
        info = {}

        guess = WordleEnv.convert_action(action)
        info["guess"] = guess
        a = WordleEnv.convert_word(self.answer)
        info["a"] = a

        if guess in WORDS:

            # Give a small reward for guessing a valid English word
            reward += VALID_WORD

            # Create an array of the 5 letters in the answer so we can track what we've seen
            letters = []
            info["letters"] = letters
            for i in range(5):
                letters.append(
                    {
                        "letter": self.answer[i],
                        "seen": False,
                        "position": i,
                        "b": a[i],
                    }
                )

            # Check each letter in the guess to see if it's correct
            for i in range(5):
                if a[i] == action[i]:
                    # The letter is in the correct location
                    observation[i] = 2
                    reward += CORRECT_LETTER
                    letters[i]["seen"] = True

            # Now check for misplaced letters
            for i in range(5):
                if a[i] != action[i] and WordleEnv.check_letter(action[i], letters):
                    # The letter is in the word but in the wrong location
                    # We check the remaining letters to prevent double counting a repeat,
                    # for example, if the word is WORLD, and the guess is LLLLL, only count index 3
                    observation[i] = 1
                    reward += MISPLACED_LETTER

        else:
            # Give a penalty for trying an invalid word
            reward += INVALID_WORD

        all_correct = True
        for i in range(5):
            if observation[i] != 2:
                all_correct = False
        if all_correct:
            done = True

        self.last_step = (action, observation, reward, done, info)

        return (observation, reward, done, info)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed, return_info=return_info, options=options)

        # Generate a new random word from the list
        words = WORDS
        self.answer = words[default_rng().integers(low=0, high=len(words))]
        self.last_step = WordleEnv.empty_step()

        observation = [0, 0, 0, 0, 0]
        if return_info:
            return observation, {}
        return observation

    def render(self, mode="human"):

        outfile = StringIO() if mode == "ansi" else sys.stdout

        # self.last_step = (action, observation, reward, done, info)
        print("last_step", self.last_step)
        action = self.last_step[0]
        guess = WordleEnv.convert_action(action)
        observation = self.last_step[1]
        blocks = [BLACK for i in range(5)]
        for i in range(5):
            o = observation[i]
            if o == 0:
                blocks[i] = BLACK
            if o == 1:
                blocks[i] = YELLOW
            if o == 2:
                blocks[i] = GREEN
        reward = self.last_step[2]

        outfile.write(f"Answer: {self.answer}\n")
        outfile.write(f"Guess:  {guess}\n")
        outfile.write(f"Clue:   {''.join(blocks)}\n")
        outfile.write(f"Reward: {reward}\n")
        outfile.write("\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
        return None

    @staticmethod
    def convert_word(word):
        "Convert a string to an array of bytes, to be used as an action"

        action = []
        for c in word:
            action.append(ord(c) - 65)
            # action.append(np.cast["ubyte"](ord(c) - 65))
        return action

    @staticmethod
    def convert_action(action):
        "Convert an action array to a string of ASCII characters"

        if len(action) != 5:
            raise Exception("action must have 5 elements")

        a = np.empty(5, dtype=np.ubyte)
        for i in range(5):
            a[i] = action[i] + 65

        return bytes(a).decode()

    @staticmethod
    def check_letter(g, letters):
        "Returns true if the guessed letter is in the word and we haven't counted it already"
        for i in range(5):
            letter = letters[i]
            if letter["b"] == g:
                if letter["seen"]:
                    return False
                letter["seen"] = True
                return True
        return False
