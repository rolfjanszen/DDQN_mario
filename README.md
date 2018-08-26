# DDQN_mario

Uses a Deep Reinforcement Learning with Double Q-learning to play super mario bros. 1
https://arxiv.org/pdf/1509.06461.pdf

Requires:
openai gym
mario gym environment: https://github.com/ppaquette/gym-super-mario
tensorflow
numpy

Run: $python3 mario_ai.py


I have not yet run this for longer periods of time.

The learning rate and epsilon are set high initially to speed up learning at the beginning. But decay at a rate of 0.01 at each training step.

