import gym
import argparse
from itertools import count

from utils import *
from visual import Visualizer
from dqn import SARS, Agent, DQN_optimizer

# MACRO
ENV = 'CartPole-v0'
CAPACITY = int(1e5)
NUM_EPISODE = int(1e3)
NUM_TEST = 10
BATCH_SIZE = 32

GAMMA = 0.9
INIT_EPS = 0.5
END_EPS = 0.01
EPS_DECAY = 200
INIT_LR = 0.001


def main(args):
    print('Playing {}'.format(args.env))
    env = gym.make(args.env).unwrapped

    num_state = env.observation_space.shape[0]
    actions = action_space_wrapper(env.action_space)

    agent = Agent(num_state, actions, args.init_eps,
                  args.end_eps, args.eps_decay)
    agent_optimizer = DQN_optimizer(
        agent, args.gamma, args.capacity, args.batch_size, args.init_lr)

    episode_length = Visualizer()

    for episode in range(args.num_episode):
        state = env.reset()

        running_loss = 0.
        for step in count():
            action = agent.e_greedy_act(state)
            # action = agent.boltzmann_act(state)

            next_state, reward, done, _ = env.step(action)
            # env.render()

            if done:
                next_state = None
            agent_optimizer.memory.push_back(
                    SARS(state, action, reward, next_state))


            if agent_optimizer.memory.trainable:
                loss = agent_optimizer.step()
                running_loss += loss

            state = next_state

            if done:
                print('[Train] The {}th episode ends at {} time step. Its average loss is {:.4f}, running loss is {:.4f}'.format(
                    episode, step, running_loss/step, running_loss))
                episode_length.push_back(step)
                break

        if episode % 100 == 99:
            total_reward = 0.

            for eval in range(NUM_TEST):
                state = env.reset()
                
                for step in count():
                    # evaluating agent greedily
                    action = agent.greedy_act(state)
                    action = action[0]

                    next_state, reward, done, _ = env.step(action)
                    env.render()

                    state = next_state
                    total_reward += reward

                    if done:
                       break
            print('[Eval] total reward: {}, average reward: {}'.format(total_reward, total_reward/NUM_TEST))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', default=ENV,
                        help='setting environment', type=str)
    parser.add_argument('--capacity', '-c', default=CAPACITY,
                        help='capacity of replay memory', type=int)
    parser.add_argument('--num_episode', '-n',
                        default=NUM_EPISODE, help='number of episode', type=int)
    parser.add_argument('--batch_size', '-b',
                        default=BATCH_SIZE, help='batch size', type=int)
    parser.add_argument('--gamma', '-g', default=GAMMA,
                        help='gamma in bellman backup', type=float)
    parser.add_argument('--init_eps', '-ie', default=INIT_EPS, help='initial epsilon', type=float)
    parser.add_argument('--end_eps', '-ee', default=END_EPS, help='end epsilon', type=float)
    parser.add_argument('--eps_decay', '-ed', default=EPS_DECAY, help='end epsilon', type=int)
    parser.add_argument('--init_lr', '-il', default=INIT_LR, help='initial learning rate', type=float)
    args = parser.parse_args()
    main(args)