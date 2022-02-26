"Wordle unit tests"

import unittest
import gym
import numpy
from gym.envs.registration import register
from wordle import WordleEnv


register(
    id="wordle-v0",
    entry_point="wordle:WordleEnv",
)


class TestWordle(unittest.TestCase):
    "Wordle Tests"

    def setUp(self):
        "Set up the test"
        self.wordle = WordleEnv()

    def test_convert_word(self):
        "Test converting an ASCII word to an array of numbers to be used as an action"

        word = "HELLO"
        action = [7, 4, 11, 11, 14]
        converted = WordleEnv.convert_word(word)

        # print("word", word)
        # print("action", action)
        # print("converted", converted)
        self.assertTrue(numpy.array_equal(action, converted))

    def test_convert_action(self):
        "Test converting an action array to a string"

        word = "HELLO"
        action = [7, 4, 11, 11, 14]
        converted = WordleEnv.convert_action(action)
        # print("word", word)
        # print("action", action)
        # print("converted", converted)
        self.assertTrue(word == converted)

    def test_check_letter(self):
        "Test checking for misplaced letters"

        letters = []
        letters.append(
            {
                "letter": "A",
                "seen": False,
                "position": 0,
                "b": 0,
            }
        )
        letters.append(
            {
                "letter": "B",
                "seen": False,
                "position": 1,
                "b": 1,
            }
        )
        letters.append(
            {
                "letter": "C",
                "seen": False,
                "position": 2,
                "b": 2,
            }
        )
        letters.append(
            {
                "letter": "D",
                "seen": False,
                "position": 3,
                "b": 3,
            }
        )
        letters.append(
            {
                "letter": "E",
                "seen": False,
                "position": 4,
                "b": 4,
            }
        )
        self.assertEqual(True, WordleEnv.check_letter(1, letters))
        self.assertEqual(False, WordleEnv.check_letter(1, letters))
        self.assertEqual(False, WordleEnv.check_letter(20, letters))

    def test_run(self):
        "Test a few steps"

        env = gym.make("wordle-v0")
        observation = env.reset()

        env.set_answer("WORLD")
        action = WordleEnv.convert_word("HELLO")
        observation, reward, done, _ = env.step(action)
        # env.render()
        self.assertEqual(observation[0], 0)
        self.assertEqual(observation[1], 0)
        self.assertEqual(observation[2], 0)
        self.assertEqual(observation[3], 2)
        self.assertEqual(observation[4], 1)
        self.assertEqual(done, False)

        env.set_answer("HELLO")
        action = WordleEnv.convert_word("WORLD")
        observation, reward, done, _ = env.step(action)
        # env.render()
        self.assertEqual(observation[0], 0)
        self.assertEqual(observation[1], 1)
        self.assertEqual(observation[2], 0)
        self.assertEqual(observation[3], 2)
        self.assertEqual(observation[4], 0)
        self.assertEqual(done, False)

        action = WordleEnv.convert_word("HWHWH")
        observation, reward, done, _ = env.step(action)
        # env.render()
        self.assertEqual(observation[0], 0)
        self.assertEqual(observation[1], 0)
        self.assertEqual(observation[2], 0)
        self.assertEqual(observation[3], 0)
        self.assertEqual(observation[4], 0)
        self.assertEqual(reward < 0, True)
        self.assertEqual(done, False)

        action = WordleEnv.convert_word("HELLO")
        observation, reward, done, _ = env.step(action)
        # env.render()
        self.assertEqual(observation[0], 2)
        self.assertEqual(observation[1], 2)
        self.assertEqual(observation[2], 2)
        self.assertEqual(observation[3], 2)
        self.assertEqual(observation[4], 2)
        self.assertEqual(reward > 0, True)
        self.assertEqual(done, True)

        env.close()


if __name__ == "__main__":
    unittest.main()
