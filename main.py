from __future__ import division
import sys

# Necessary to load correct packages - circumvent this by virtualenv in the future
sys.path.append( '/Users/anthony/anaconda/lib/python2.7/site-packages/' )
sys.path.remove( '/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python' )

import numpy as np
import tensorflow as tf
import gym
import itertools
from agent import Agent
from rm import ReplayMemory
from utils import hard_update


def run_experiment( sess, hps ):
    hps['env'] = 'Pendulum-v0'
    # hps['env'] = 'BipedalWalker-v2'
    env = gym.make( hps['env'] )
    hps['s_dim'] = env.observation_space.shape[0]
    hps['a_dim'] = env.action_space.high.shape[0]
    hps['a_bound'] = env.action_space.high
    # TODO: Assert action symmetry

    rm = ReplayMemory( hps['buffer_size'], hps['batch_size'] )
    agent = Agent( sess, hps, rm )

    sess.run( tf.global_variables_initializer() )
    do_hard_update( sess, 'actor_target', 'actor' )
    do_hard_update( sess, 'critic_target', 'critic' )

    for i in range( 0, hps['num_episodes'] ):
        s1 = env.reset()
        agent.ou.reset()
        ep_reward = 0
        for t in itertools.count():
            # if i % hps['render_every'] == 0: env.render()
            # if i > 100: env.render()
            env.render()
            a1 = agent.explore( np.reshape( s1, ( 1, hps['s_dim'] ) ), i )
            s2, r1, d, _ = env.step( np.reshape( a1, ( hps['a_dim'], ) ) )
            if not d:
                rm.add( np.reshape( s1, ( hps['s_dim'], ) ),
                        np.reshape( a1, ( hps['a_dim'], ) ),
                        np.reshape( r1, ( 1, ) ),
                        np.reshape( s2, ( hps['s_dim'], ) ) )
            if rm.size() > hps['batch_size']:
                agent.learn()
            s1 = s2
            ep_reward += r1
            if d:
                print( '| Reward: {:d} | Episode: {:d} | Length: {:d}'
                    .format( int(ep_reward), i, t + 1 ) )
                break


def main():
    hps = {
        'render_every': 30,
        'num_episodes': 10000,
        'buffer_size': 100000,
        'batch_size': 64,
        'noise_decay': 0.9999,
        'actor_lr': 0.0001,
        'critic_lr': 0.001,
        'tau': 0.001,
        'gamma': 0.99,
        'h1_actor': 400,
        'h2_actor': 300,
        'h3_actor': 300,
        'h1_critic': 400,
        'h2_critic': 400,
        'h3_critic': 300,
        'l2_reg_actor': 1e-6,
        'l2_reg_critic': 1e-6 }

    tf.reset_default_graph()
    with tf.Session() as sess:
        run_experiment( sess, hps )


if __name__ == '__main__':
    main()
