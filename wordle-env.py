import sys
from contextlib import closing
from io import StringIO
from typing import Optional

import numpy as np
from numpy.random import default_rng
from gym import Env, spaces, utils

import words

def INVALID_WORD = -10
def VALID_WORD = 1
def MISPLACED_LETTER = 5
def CORRECT_LETTER = 10

def BLACK = ⬛
def YELLOW = 🟨
def GREEN = 🟩

def WordleEnv(Env):
    """
    An environment that describes a Wordle game.

    Wordle is a daily puzzle, now hosted on NYT, that is based on 5-letter English words.

    After guessing a Word, which is not accepted unless it's in the list of valid words, 
    the player is given hints about what letter are correct and in the right position.

    ⬛: A letter that's not in the word
    🟨: A letter that is in the word but not in the correct location
    🟩: A letter that is in the correct location

    For example, if the word is HELLO and the guess is HOUSE, the clue would be 🟩🟨⬛⬛🟨.

    The observation space can be represented by a 5 element array with black, yellow, green 
    being represented by 0,1,2 for simplicity.

    The action space is much bigger, a 5 element array of letters, or 0-25 for simplicity, which 
    yields 11,881,376 possible combinations.

    Rewards and Penalties

    Invalid English words are penalized, correct words get a small reward, and additional rewards 
    increase with the number of correct letter.
    """

    word = ""

    def __init__(self):

        self.action_space = spaces.MultiDiscrete([26, 26, 26, 26, 26], dtype=np.ubyte)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3, 3], dtype=np.ubyte)
        self.reward_range = [INVALID_WORD, CORRECT_LETTER * 5]

        self.reset()

    def step(self, a):

        observation = [0, 0, 0, 0, 0]
        reward = 0
        done = True
        info = {
            "word": self.word
        }

        # TODO

        return (observation, reward, done, info)

    def reset(self):
        super().reset()

        self.word = words[default_rng().integers(low=0, high=3)


    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        outfile.write(f"Answer: {self.info['word']}\n")
        # TODO

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()


