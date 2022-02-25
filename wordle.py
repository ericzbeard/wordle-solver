"""
Wordle Gym Environment
"""

import sys
from contextlib import closing
from io import StringIO
from typing import Optional

import numpy as np
from numpy.random import default_rng
from gym import Env, spaces, utils

import all_words

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

    answer = ""

    def __init__(self):

        self.action_space = spaces.MultiDiscrete([26, 26, 26, 26, 26], dtype=np.ubyte)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3, 3], dtype=np.ubyte)
        self.reward_range = [INVALID_WORD, CORRECT_LETTER * 5]

        self.last_step = None

        self.reset()

    def step(self, action):

        observation = [0, 0, 0, 0, 0]
        reward = 0
        done = True
        info = {}

        # TODO

        self.last_step = (action, observation, reward, done, info)
        return (observation, reward, done, info)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset()

        self.answer = all_words.WORDS[default_rng().integers(low=0, high=3)]

    def render(self, mode="human"):

        outfile = StringIO() if mode == "ansi" else sys.stdout

        # self.last_step = (action, observation, reward, done, info)
        action = self.last_step[0]
        observation = self.last_step[1]
        reward = self.last_step[2]
        done = self.last_step[3]
        info = self.last_step[4]

        outfile.write(f"Answer: {self.answer}\n")
        outfile.write(f"Clue:   {observation}\n")
        outfile.write(f"Reward: {reward}\n")
        # TODO

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
        return None
