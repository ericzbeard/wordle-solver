# Wordle Solver

This is a study project to experiment with reinforcement learning to solve Wordle. 
Obviously, using machine learning is not the most efficient way to play Wordle with code, 
but it makes for a good example of a problem that _might_ be solvable with an RL algorithm.

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

My initial research led me to [OpenAI Gym](https://github.com/openai/gym), which is 
a framework for describing the problem that you want to solve. Check out the 
`WordleEnv` class in `wordle.py` to see how the environment is set up. There's no 
ML involved here yet, it's just a way for the learning algorithm to get the data it 
needs to train the model.

```Python
observation, reward, done, info = env.step(action)
```

That line of code is the primary function of an environment. Your algorithm makes an action, 
in this case an array of letters, and it receives in return a observation of the environment ater that action. It also gets a reward or a penalty, depending on how well it did.

## Frameworks

I thought `rl_coach` would be a good idea, since it is used in Sagemaker tutorials, but I 
found that it requires an older version of gym, which causes my environment to fail. And it 
does not support `MultiDiscrete`: [https://github.com/IntelLabs/coach/issues/176](https://github.com/IntelLabs/coach/issues/176)

AWS supports [Tensorflow](https://www.tensorflow.org/), MXNet, and Pytorch. Which one to use? They all seem to have pros and cons but from a few searches, it's obvious that Tensorflow has the best documentation, examples, and community support. My goal here is to find an algorithm that is already implemented and "simply" plug it in to my environment. It didn't take much reading to see that [Keras](https://keras.io/) is a popular high level API for interacting with Tensorflow. And tensorflow can run models in the browser, which might make it easier to implement a demo for my Wordle solver.

## Algorithms

There are many RL algorithms to choose from, but I think a DQN (Deep Q Network) is what I want. It is commonly used in RL tutorials as a way to play Atari games, which are not an exact analogy to Wordle, but one important commonality is that the model needs a memory of past states and actions. I need the algorithm to know that if it got yellows or greens on prior tries, it should re-use those letters. And I also need it to develop a sense for good starting words.

DQN: didn't work, due to large action space
Actor-Critic: didn't work, due to predictions being floats instead of bytes in range 0-26
A2C: ? Very fast but converges on a single word after finally seeing a reward

Maybe what I need to do is split the problem into two problems.
1. Train a model to learn english words. 
2. Plug those words into the second model, which plays the game.

Hard to do 1 without cheating though. I could build a custom environment that would help 
it learn english, but how different is that from simply sampling from known words?

Maybe learning needs 2 phases, with different hyperparameters set. Set the learning to be 
very slow at first, so it tries lot of random samples. Then learn faster once it starts 
making accurate guesses.

Using random digits, it will take an average of 18,000 guesses to get a single reward.
And then probably some multiple of 11M experiments to start to understand which words 
are valid. Once that training is done, that model could be used as the random sample, instead 
of a truly random sample.

## Tutorials and Documentation to read

I found [this](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/) to be a good introduction to Gym, but it uses a pre-existing environment. There are not many detailed tutorials on how to actually create an Environment from scratch, so I spent some time reading the source code for [Taxi-v3](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py).

The documentation at [https://www.gymlibrary.ml/](https://www.gymlibrary.ml/) is a good start, but the site at [https://gym.openai.com/](https://gym.openai.com/) is nearly useless.

- [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/)
- [Reinforcement Learning w/ Keras + OpenAI: DQNs](https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c)
- [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
- [Tensorflow RL Tutorial](https://www.tensorflow.org/agents/tutorials/0_intro_rl)
- [Reinforcement Learning algorithms — an intuitive overview](https://smartlabai.medium.com/reinforcement-learning-algorithms-an-intuitive-overview-904e2dff5bbc)





