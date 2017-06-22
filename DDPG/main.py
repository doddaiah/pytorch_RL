import gym
from gym import wrappers
import argparse
import numpy as np
from itertools import count

from ddpg import SARS, DDPGAgent, DDPGOptimizer

ENV = 'BipedalWalker-v2'
CAPACITY = int(1e6)
NUM_EPISODE = int(1e4)
NUM_TEST = 5
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 1e-3
WEIGHT_DECAY = 1e-2
ACTOR_INIT_LR = 1e-4
CRITIC_INIT_LR = 1e-3
NUMPY_PRECISION = np.float32


def main(args):
    env = gym.make(args.env)
    outdir = '/tmp/ddpg'
    env = wrappers.Monitor(env, outdir, force=True)

    assert (env.action_space.high == -env.action_space.low).all(), 'action_space bound should be symmetric'
    assert (env.action_space.high == env.action_space.high[0]).all(), 'all action dims should have the same bound'

    agent = DDPGAgent(env.observation_space.shape[0],
                      env.action_space.shape[0],
                      float(env.action_space.high[0]))
    optimizer = DDPGOptimizer(agent, args.capacity, args.batch_size,
                              args.gamma, args.tau, args.init_lr, args.weight_decay)

    for episode in range(args.num_episode):
        agent.ou_noise.reset()
        state = env.reset().astype(NUMPY_PRECISION)

        running_loss = 0.
        training_total_reward = 0.
        for step in count():
            action = agent.noisy_act(state)

            next_state, reward, done, _ = env.step(action)
            # env.render()

            state, action, reward, next_state = map(
                lambda x: NUMPY_PRECISION(x), (state, action, reward, next_state))

            if done:
                next_state = None
            optimizer.memory.push_back(
                SARS(state, action, reward, next_state))

            if optimizer.memory.trainable:
                loss = optimizer.step()
                running_loss += loss

            state = next_state
            training_total_reward += reward

            if done:
                optimizer.stats.add_scalar_value('average loss', running_loss/step)
                optimizer.stats.add_scalar_value('step', step)
                optimizer.stats.add_scalar_value('training total reward',
                        training_total_reward)
                break

        if episode % 100 == 99:
            total_reward = 0.

            for eval in range(args.num_test):
                # agent.ou_noise.reset()
                state = env.reset().astype(NUMPY_PRECISION)

                for step in count():
                    # action = agent.noisy_act(state)
                    action = agent.act(state)
                    # print(action)

                    next_state, reward, done, _ = env.step(action)
                    env.render()

                    state, action, reward, next_state = map(
                        lambda x: NUMPY_PRECISION(x), (state, action, reward, next_state))

                    state = next_state
                    total_reward += reward

                    if done:
                        break
            print('[Eval] {}th episode, total reward: {}, average reward: {}'.format(
                episode, total_reward, total_reward/args.num_test))
            
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', default=ENV,
                        help='setting environment', type=str)
    parser.add_argument('--capacity', '-c', default=CAPACITY,
                        help='capacity of replay memory', type=int)
    parser.add_argument('--num_episode',
                        default=NUM_EPISODE, help='number of episode', type=int)
    parser.add_argument('--num_test',
                        default=NUM_TEST, help='number of test', type=int)
    parser.add_argument('--batch_size', '-b',
                        default=BATCH_SIZE, help='batch size', type=int)
    parser.add_argument('--init_lr', '-il', default={'actor': ACTOR_INIT_LR, 'critic': CRITIC_INIT_LR},
                        help='initial learning rate', type=float)
    parser.add_argument('--gamma', '-g', default=GAMMA,
                        help='gamma in bellman backup', type=float)
    parser.add_argument('--tau', default=TAU,
                        help='tau in updating target', type=float)
    parser.add_argument('--weight_decay', default=WEIGHT_DECAY,
                        help='weight decay in optimizing', type=float)
    args = parser.parse_args()
    main(args)
