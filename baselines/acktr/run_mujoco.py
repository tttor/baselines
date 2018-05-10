#!/usr/bin/env python3

import os
import time
import pickle
import datetime
import socket

import git
import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser

from baselines.acktr import acktr_cont
from baselines.acktr.actor_net import GaussianMlpPolicy
from baselines.acktr.critic_net import NeuralNetValueFunction
from baselines.acktr.filters import ZFilter

def main():
    hostname = socket.gethostname(); hostname = hostname.split('.')[0]
    stamp = datetime.datetime.now().strftime("acktr-reacher-"+hostname+"-%Y%m%d-%H%M%S-%f")
    xprmt_dir = os.path.join(os.path.expanduser("~"),'xprmt/acktr-reacher')
    xprmt_dir = os.path.join(xprmt_dir,stamp)

    args = mujoco_arg_parser().parse_args()
    logger.configure(dir=xprmt_dir)
    repo = git.Repo(search_parent_directories=True)
    csha = repo.head.object.hexsha
    ctime = time.asctime(time.localtime(repo.head.object.committed_date))
    cmsg = repo.head.object.message.strip()
    logger.log('gitCommitSha= %s'%csha)
    logger.log('gitCommitTime= %s'%ctime)
    logger.log('gitCommitMsg= %s'%cmsg)

    train(args, xprmt_dir)

def train(args, xprmt_dir):
    with tf.Session(config=tf.ConfigProto()) as sess:
        env = make_mujoco_env(args.env, args.seed)
        obfilter = ZFilter(env.observation_space.shape)

        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        print('ob_dim= '+str(ob_dim))
        print('ac_dim= '+str(ac_dim))
        print('env.spec.timestep_limit= '+str(env.spec.timestep_limit))

        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            pi = GaussianMlpPolicy(ob_dim, ac_dim)

        saver = tf.train.Saver()

        ## train offline
        acktr_cont.learn(env,
                         policy=pi, vf=vf,
                         rollout=run_one_episode, obfilter=obfilter,
                         gamma=0.99, lam=0.97,
                         batch_size=2500,# in nsteps
                         max_nsteps=args.num_timesteps,
                         desired_kl=0.002,
                         animate=False)

        saver.save(sess, os.path.join(xprmt_dir,'training_acktr_reacher'))

        ## test just after training
        neps = 100
        paths = []
        print("***** testing *****")
        for ep_idx in range(neps):
            path = acktr_cont.run_one_episode(env, pi, obfilter, render=False)
            paths.append(path)

        logger.record_tabular("TestingNEp", len(paths))
        logger.record_tabular("TestingEpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("TestingEpLenMean", np.mean([path["length"] for path in paths]))
        logger.record_tabular("Seed", args.seed)
        logger.dump_tabular()

        with open(os.path.join(xprmt_dir,'obfilter.pkl'), 'wb') as f: pickle.dump(obfilter, f)
        env.close()

if __name__ == "__main__":
    main()
