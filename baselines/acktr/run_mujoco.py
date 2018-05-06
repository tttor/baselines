#!/usr/bin/env python3

import tensorflow as tf

from baselines import logger
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser

from baselines.acktr import acktr_cont
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction

def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()

    env = make_mujoco_env(args.env, args.seed)

    ## train
    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        print('ob_dim= '+str(ob_dim))
        print('ac_dim= '+str(ac_dim))
        print('env.spec.timestep_limit= '+str(env.spec.timestep_limit))

        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        acktr_cont.learn(env,
                         policy=policy, vf=vf,
                         gamma=0.99, lam=0.97,
                         batch_size=2500,# in nsteps
                         max_nsteps=args.num_timesteps,
                         desired_kl=0.002,
                         animate=False)

    env.close()

if __name__ == "__main__":
    main()
