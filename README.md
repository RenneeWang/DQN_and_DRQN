# DQN_and_DRQN
Course project for CIE6032 group 26

## Require Environment
* Python >= 3.6
* Gym\[Atari]
* Gym\[Box2D]
* PyTorch

## Abstract
Deep Reinforcement Learning (RL) has achieved promising results at training AIs to play games in OpenAI Gym Library. However, current algorithms are facing challenges when playing games like Skiing due to the Partial Observed Markov Decision Process (POMDP). This problem is investigated, and the RL model with Recurrent Neural Network (RNN) architecture is used to eliminate the effects caused by reward evaluation standards with long-term dependencies. Both Skiing and Lunar Lander games are examined in this project to verify the expectation. Experiments show that compared with Deep Q Network (DQN), deep recurrent Q-network (DRQN) can improve the performance of agents on Skiing game to a certain extent. Meanwhile, the improvement effect of DRQN on the Lunar Lander agent is not obvious due to less correlation between final reward and previous states.

## Note
* Use Training Result.ipynb to load the weight and see the training result
* train_ram.py can train the model
