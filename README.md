# Wordle Solver

This is a study project to experiment with reinforcement learning to solve Wordle. 
Obviously, using machine learning is not the most efficient way to play Wordle with code, 
but it makes for a good example of a problem that _can_ be solved with an RL algorithm.

I'm studying for the AWS AI/ML exam, and my preference is always to build something rather than 
spend all day memorizing documentation.

*The Rules*:

* I can’t pre-program English words into the system
* I can’t program the meaning of black-yellow-green into the system
* I can’t look at any other similar solutions online

All the algorithm will know is that it needs to enter 5 random numbers from 0 to 25 
(which are converted to ASCII upper case latters), and that it gets back an observation 
that consists of a set of 5 numbers, along with a reward that tells it how well it did. 
It will have to basically learn English and then learn the rules of the game in whatever 
magical way that the existing RL algorithms work.

The plan is not write a new algorithm, just the custom environment and the infrastructure that 
it runs on to train the model and then make it available for inference, using AWS Sagemaker.

To demonstrate the trained model, the endpoint API would look something like

`function guess(letters): result`

Where result is an array that contains what’s currently known about the word.
For example, a new puzzle would be [0, 0, 0, 0, 0] (⬛⬛⬛⬛⬛), A puzzle
that had the second letter correct would be [0, 2, 0, 0, 0] (⬛🟩⬛⬛⬛) and a
puzzle with the third letter correct but out of place would be [0, 0, 1, 0, 0]
(⬛⬛🟨⬛⬛). The model should be able to formulate the next guess based on any
state. Possible number of states are 3^5 = 243. 

There are 11,881,376 possible guesses, without any knowledge of English.

My initial research led me to (https://github.com/openai/gym)[OpenAI Gym], which is 
a framework for describing the problem that you want to solve. Check out the 
`WordleEnv` class in `wordle.py` to see how the environment is set up. There's no 
ML involved here yet, it's just a way for the learning algorithm to get the data it 
needs to train the model.

```Python
observation, reward, done, info = env.step(action)
````

That line of code is the primary function of an environment. Your algorithm makes an action, 
in this case an array of letters, and it receives in return a observation of the environment ater that
action. It also gets a reward or a penalty, depending on how well it did.

