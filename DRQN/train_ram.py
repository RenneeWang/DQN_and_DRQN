import numpy as np
import gym
from utils import *
from agent import *
from config import *
import pandas as pd
from NORM import normalized


def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t,r):
    rewards_log = []
    average_log = []
#     memory_log = []
    eps = eps_init

    for i in range(1, 1 + num_episode):

        episodic_reward = 0
        done = False
        state = env.reset()
        state = normalized(state)
        t = 0
        
        

        while not done and t < max_t:

            t += 1
            action = agent.act(torch.FloatTensor(np.expand_dims(np.expand_dims(state, 0), 0)), eps)
#             print(type(action))
            next_state, reward, done, _ = env.step(action)
            next_state = normalized(next_state)
            agent.memory.append((state, action, reward, next_state, done))
#             memory_log.append((i, t, state, action, reward, next_state, done))

            if t % 4 == 0 and len(agent.memory) >= agent.bs:
                agent.learn()
                agent.soft_update(agent.tau)

            state = next_state.copy()
            episodic_reward += reward

        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episodic_reward, average_log[-1]), end='')
        if i % 50 == 0:
            print()

        eps = max(eps * eps_decay, eps_min)
        
#     memorys = pd.DataFrame(memory_log, columns=['episode','t','state','action','reward','next_state','done'])
#     memorys.to_csv('./skiing_results/memorys_{}_{}.csv'.format(RAM_ENV,r), encoding='utf-8', index=0)


    return rewards_log


if __name__ == '__main__':
    rounds = 5
    env = gym.make(RAM_ENV)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE, False)
#     agent.Q_local.load_state_dict(torch.load('Skiing-ramDeterministic-v4_weights.pth'))
#     agent.Q_target.load_state_dict(torch.load('Skiing-ramDeterministic-v4_weights.pth'))
    for r in range(rounds):
        print("training round {}".format(r))
        rewards_log = train(env, agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T,r)
        np.save('./skiing_results/{}_r{}_rewards.npy'.format(RAM_ENV,r), rewards_log)
        agent.Q_local.to('cpu')
        torch.save(agent.Q_local.state_dict(), './skiing_results/{}_r{}_weights.pth'.format(RAM_ENV,r))
        print("round {} information saved.".format(r))
        agent.Q_local.to(DEVICE)