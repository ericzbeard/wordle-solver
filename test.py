import unittest
from wordle import WordleEnv
import numpy

class TestWordle(unittest.TestCase):
    "Wordle Tests"

    def setUp(self):
        "Set up the test"
        self.wordle = WordleEnv()

    def test_convert_word(self):
        "Test converting an ASCII word to an array of numbers to be used as an action"

        word = "HELLO"
        action = [72, 69, 76, 76, 79]
        converted = WordleEnv.convert_word(word)
        self.assertTrue(numpy.array_equal(action, converted))

    def test_convert_action(self):
        "Test converting an action array to a string"

        word = "HELLO"
        action = [72, 69, 76, 76, 79]
        converted = WordleEnv.convert_action(action)
        self.assertTrue(word == converted)

if __name__ == "__main__":
    unittest.main()

