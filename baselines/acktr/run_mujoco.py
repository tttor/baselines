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
from baselines.common.cmd_util import make_mujoco_env, arg_parser
from baselines.common.filters import ZFilter
from baselines.acktr import acktr_cont
from baselines.acktr.actor_net import GaussianMlpPolicy
from baselines.acktr.critic_net import NeuralNetValueFunction

home_dir = os.path.expanduser("~")
asset_dir = os.path.join(home_dir, 'ws/gym/gym/envs/mujoco/assets')

def main():
    args = acktr_arg_parser().parse_args()
    repo = git.Repo(search_parent_directories=True)
    csha = repo.head.object.hexsha
    ctime = time.asctime(time.localtime(repo.head.object.committed_date))
    cmsg = repo.head.object.message.strip()
    hostname = socket.gethostname(); hostname = hostname.split('.')[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stamp = '_'.join(['acktr',args.mode,args.env,hostname,timestamp])
    xprmt_dir = os.path.join(args.dir, stamp)
    if args.mode=='train':
        assert args.nsteps is not None
    elif args.mode=='test':
        assert args.neps is not None
        delim = '/xprmt'; ckpt_lines = []
        ckpt_fpath_src = os.path.join(args.dir, 'checkpoint')
        with open(ckpt_fpath_src, 'r') as f:
            for line in f:
                k, v = line.strip().split(':')
                ckpt_home_dir, part = [i.strip('"') for i in v.strip().split(delim)]
                if ckpt_home_dir != home_dir:
                    ckpt_line = k+': '+'"'+os.path.join(home_dir,'xprmt',part)+'"'
                    ckpt_line = k+': '+'"'+home_dir+delim+part+'"'
                    ckpt_lines.append(ckpt_line)
        if len(ckpt_lines) > 0:
            ckpt_fpath_dst = os.path.join(args.dir, 'checkpoint.ori')
            os.rename(ckpt_fpath_src, ckpt_fpath_dst)
            with open(ckpt_fpath_src, 'w') as f:
                for l in ckpt_lines: f.write(l+'\n')
    else:
        assert False, 'fatal: unknown mode!!!'
    logger.configure(dir=xprmt_dir)
    logger.log('gitCommitSha= %s'%csha)
    logger.log('gitCommitTime= %s'%ctime)
    logger.log('gitCommitMsg= %s'%cmsg)
    logger.log('seed= %i'%args.seed)

    ## prepare model xml with the correct timestep
    env_id, timestep = args.env.split('@')
    if args.mode=='test': assert env_id in args.dir
    bare_env_id = env_id.lower().replace('-v2','')
    xml_src = os.path.join(asset_dir,bare_env_id,bare_env_id+str('.xml')+'@'+timestep)
    xml_dst = os.path.join(asset_dir,bare_env_id+str('.xml'))
    try: os.remove(xml_dst)
    except OSError: pass
    os.symlink(xml_src, xml_dst)

    ## run!
    env = make_mujoco_env(env_id, args.seed)
    os.remove(xml_dst)
    print('***** env: created! *****')
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

        ## train offline
        acktr_cont.learn(env,
                         policy=pi, vf=vf,
                         rollout=acktr_cont.run_one_episode, obfilter=obfilter,
                         gamma=0.99, lam=0.97,
                         batch_size=2500,# in nsteps
                         max_nsteps=nsteps,
                         desired_kl=0.002,
                         animate=False)

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(xprmt_dir,'training_acktr_reacher'))

        ## test just after training
        neps = 100
        paths = []
        logger.log("***** immediate testing *****")
        for ep_idx in range(neps):
            path = acktr_cont.run_one_episode(env, pi, obfilter, render=False)
            paths.append(path)

        logger.record_tabular("TestingNEps", len(paths))
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
            meta_graph.restore( sess,tf.train.latest_checkpoint(checkpoint_dir=xprmt_dir) )
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
            logger.record_tabular("ReachingStepLen", path['reaching_step_len'])
            logger.dump_tabular()

        logger.record_tabular("EpRetMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("EpLenMean", np.mean([path["length"] for path in paths]))
        logger.record_tabular("EpReachingStepLenMean", np.mean([path["reaching_step_len"] for path in paths]))
        logger.record_tabular("NEps", neps)
        logger.dump_tabular()

def acktr_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--mode', help='mode', type=str, choices=['train','test'], default=None, required=True)
    parser.add_argument('--env', help='environment ID', type=str, default=None, required=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None, required=True)
    parser.add_argument('--dir', help='(xprmt) dir', type=str, default=None, required=True)
    parser.add_argument('--nsteps', help='nsteps', type=int, default=None)
    parser.add_argument('--neps', help='num of episodes', type=int, default=None)
    return parser

if __name__ == "__main__":
    main()
