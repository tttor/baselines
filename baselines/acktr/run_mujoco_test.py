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
from baselines.acktr.run_mujoco import run_one_episode

def main():
    hostname = socket.gethostname(); hostname = hostname.split('.')[0]
    stamp = datetime.datetime.now().strftime("acktr-reacher-test-"+hostname+"-%Y%m%d-%H%M%S-%f")
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

    test(args)

def test(args):
    neps = 100
    xprmt_dir = '/home/tor/xprmt/acktr-reacher/acktr-reacher-goliath-20180508-191258-145074'
    meta_fpath = os.path.join(xprmt_dir,'training_acktr_reacher.meta')
    meta_graph = tf.train.import_meta_graph(meta_fpath)

    with tf.Session(config=tf.ConfigProto()) as sess:
        env = make_mujoco_env(args.env, args.seed)
        with open(os.path.join(xprmt_dir,'obfilter.pkl'), 'rb') as f:
            obfilter = pickle.load(f)

        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]

        with tf.variable_scope("pi"):
            pi = GaussianMlpPolicy(ob_dim, ac_dim)

        meta_graph.restore( sess,tf.train.latest_checkpoint(xprmt_dir) )
        graph = tf.get_default_graph()

        paths = []
        for ep_idx in range(neps):
            path = run_one_episode(env, pi, obfilter, render=False)
            paths.append(path)

        logger.record_tabular("TestingEpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("TestingEpLenMean", np.mean([path["length"] for path in paths]))
        logger.record_tabular("TestingNEp", neps)
        logger.dump_tabular()

        env.close()

if __name__ == "__main__":
    main()
