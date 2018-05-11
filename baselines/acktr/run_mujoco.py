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
from baselines.acktr.actor_net import GaussianMlpPolicy
from baselines.acktr.critic_net import NeuralNetValueFunction
from baselines.acktr.filters import ZFilter

asset_dir = os.path.join(os.path.expanduser("~"),'ws/gym@tttor/gym/envs/mujoco/assets')

def main():
    args = mujoco_arg_parser().parse_args()
    repo = git.Repo(search_parent_directories=True)
    csha = repo.head.object.hexsha
    ctime = time.asctime(time.localtime(repo.head.object.committed_date))
    cmsg = repo.head.object.message.strip()
    hostname = socket.gethostname(); hostname = hostname.split('.')[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stamp = '_'.join(['acktr',args.mode,args.env,hostname,timestamp])
    if args.mode=='train':
        assert args.dir is None
        assert args.nsteps is not None
        xprmt_dir = os.path.join(os.path.expanduser("~"),'xprmt/acktr', stamp)
    else:
        assert args.dir is not None
        assert args.neps is not None
        xprmt_dir = os.path.join(args.dir, stamp)
    logger.configure(dir=xprmt_dir)
    logger.log('gitCommitSha= %s'%csha)
    logger.log('gitCommitTime= %s'%ctime)
    logger.log('gitCommitMsg= %s'%cmsg)
    logger.log('seed= %i'%args.seed)

    env_id, timestep = args.env.split('@')
    bare_env_id = env_id.lower().replace('-v2','')
    xml_src = os.path.join(asset_dir,bare_env_id,bare_env_id+str('.xml')+'@'+timestep)
    xml_dst = os.path.join(asset_dir,bare_env_id+str('.xml'))
    os.symlink(xml_src, xml_dst)

    env = make_mujoco_env(env_id, args.seed)
    os.remove(xml_dst)
    print('*** env: created! ***')

    if args.mode=='train':
        train(env, args.nsteps, xprmt_dir)
    else:
        test(env, args.neps, args.dir)

    env.close()

def train(env, nsteps, xprmt_dir):
    with tf.Session(config=tf.ConfigProto()) as sess:
        obfilter = ZFilter(env.observation_space.shape)
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]

        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            pi = GaussianMlpPolicy(ob_dim, ac_dim)

        saver = tf.train.Saver()

        ## train offline
        acktr_cont.learn(env,
                         policy=pi, vf=vf,
                         rollout=acktr_cont.run_one_episode, obfilter=obfilter,
                         gamma=0.99, lam=0.97,
                         batch_size=2500,# in nsteps
                         max_nsteps=nsteps,
                         desired_kl=0.002,
                         animate=False)

        saver.save(sess, os.path.join(xprmt_dir,'training_acktr_reacher'))

        ## test just after training
        neps = 100
        paths = []
        print("***** immediate testing *****")
        for ep_idx in range(neps):
            path = acktr_cont.run_one_episode(env, pi, obfilter, render=False)
            paths.append(path)

        logger.record_tabular("TestingNEp", len(paths))
        logger.record_tabular("TestingEpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("TestingEpLenMean", np.mean([path["length"] for path in paths]))
        logger.dump_tabular()

        with open(os.path.join(xprmt_dir,'obfilter.pkl'), 'wb') as f:
            pickle.dump(obfilter, f)

def test(env, neps, xprmt_dir):
    meta_graph = tf.train.import_meta_graph( os.path.join(xprmt_dir,'training_acktr_reacher.meta') )
    with tf.Session(config=tf.ConfigProto()) as sess:
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
        logger.dump_tabular()

if __name__ == "__main__":
    main()
