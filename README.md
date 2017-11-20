# Deep Deterministic Policy Gradient (DDPG)
Implementation of Lillicrap et al. (2015)'s DDPG algorithm in Tensorflow

## How to run
- To run the algorithm on a toy problem, just do `python main.py`. 
- By default it will train on OpenAI Gym's Pendulum-v0 environment and render every episode. 
- It should take <100 episodes of training for satisfactory performance. For more robust performance, change noise decay to a close to but smaller than 1.

## References
- Lillicrap, T. et al. (2015). Continuous Control with Deep Reinforcement Learning. arXiv preprint arXiv:1509.02971, 2015.
