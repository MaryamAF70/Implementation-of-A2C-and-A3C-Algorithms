# Implementation-of-A2C-and-A3C-Algorithms
This project implements and compares the Advantage Actor–Critic (A2C) and Asynchronous Advantage Actor–Critic (A3C) algorithms using the CartPole-v1 environment from OpenAI Gym . The focus is on understanding the difference between synchronous and asynchronous policy gradient methods and evaluating their performance in discrete environments.
# Environment: CartPole-v1 (Discrete Control)
The CartPole environment is one of the most classic control tasks in reinforcement learning.
A pole is attached by an un-actuated joint to a cart that moves along a track.
The goal of the agent is to prevent the pole from falling by moving the cart left or right.

Environment Specifications
Component	Description
State Space	4 continuous variables: cart position, cart velocity, pole angle, and pole angular velocity
Action Space	2 discrete actions: move left or right
Reward	+1 for every time step that the pole remains upright
Termination	Episode ends when the pole angle exceeds the threshold or the cart leaves the allowed range

The simplicity of this environment makes it ideal for testing and benchmarking actor–critic algorithms.
