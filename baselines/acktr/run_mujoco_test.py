#!/usr/bin/env python3

import os
import sys
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
from baselines.acktr.actor_net_test import GaussianMlpPolicy
from baselines.acktr.critic_net import NeuralNetValueFunction
from baselines.acktr.filters import ZFilter

def main():
    args = mujoco_arg_parser().parse_args()
    if args.neps is None: assert args.nsteps is not None
    if args.nsteps is None: assert args.neps is not None
    assert (args.neps is None) or (args.nsteps is None)

    asset_dir = '/home/tor/ws-fork/gym@tttor/gym/envs/mujoco/assets'
    env_id, timestep = args.env.split('@')
    bare_env_id = env_id.lower().replace('-v2','')
    xml_src = os.path.join(asset_dir,bare_env_id,bare_env_id+str('.xml')+'@'+timestep)
    xml_dst = os.path.join(asset_dir,bare_env_id+str('.xml'))
    try:
        os.remove(xml_dst)
    except OSError:
        pass
    os.symlink(xml_src, xml_dst)

    test(env_id, args.seed, args.neps, xprmt_dir=args.dir)

def test(env_id, seed, neps, xprmt_dir):
    hostname = socket.gethostname(); hostname = hostname.split('.')[0]
    stamp = datetime.datetime.now().strftime("test-"+hostname+"-%Y%m%d-%H%M%S-%f")
    logger.configure(dir=os.path.join(xprmt_dir, stamp))
    repo = git.Repo(search_parent_directories=True)
    csha = repo.head.object.hexsha
    ctime = time.asctime(time.localtime(repo.head.object.committed_date))
    cmsg = repo.head.object.message.strip()
    logger.log('gitCommitSha= %s'%csha)
    logger.log('gitCommitTime= %s'%ctime)
    logger.log('gitCommitMsg= %s'%cmsg)

    meta_graph = tf.train.import_meta_graph( os.path.join(xprmt_dir,'training_acktr_reacher.meta') )

    with tf.Session(config=tf.ConfigProto()) as sess:
        env = make_mujoco_env(env_id, seed)
        print('*** env: created! ***')

        with open(os.path.join(xprmt_dir,'obfilter.pkl'), 'rb') as f:
            obfilter = pickle.load(f)

        with tf.variable_scope("pi"):
            meta_graph.restore( sess,tf.train.latest_checkpoint(xprmt_dir) )
            graph = tf.get_default_graph()

            ob_dim = env.observation_space.shape[0]
            ac_dim = env.action_space.shape[0]
            pi = GaussianMlpPolicy(ob_dim, ac_dim, graph)

        paths = []
        for ep_idx in range(neps):
            path = acktr_cont.run_one_episode(env, pi, obfilter, render=True)
            paths.append(path)

            logger.record_tabular("EpReturn", path["reward"].sum())
            logger.record_tabular("EpLen", path["length"])
            logger.record_tabular("EpIdx", ep_idx)
            logger.record_tabular("ReachedAtStepIdx", path['reached_at_step_idx'])
            logger.dump_tabular()

        logger.record_tabular("TestingEpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("TestingEpLenMean", np.mean([path["length"] for path in paths]))
        logger.record_tabular("TestingEpReachedAtStepIdxMean", np.mean([path["reached_at_step_idx"] for path in paths]))
        logger.record_tabular("TestingNEp", neps)
        logger.record_tabular("TestingSeed", seed)
        logger.dump_tabular()
        env.close()

if __name__ == "__main__":
    main()
