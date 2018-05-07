#!/usr/bin/env python3

import os
import time
import pickle

import git
import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser

from baselines.acktr import acktr_cont
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
from baselines.acktr.filters import ZFilter

xprmt_dir = os.path.join(os.path.expanduser("~"),'xprmt/acktr-reacher')

def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()

    repo = git.Repo(search_parent_directories=True)
    csha = repo.head.object.hexsha
    ctime = time.asctime(time.localtime(repo.head.object.committed_date))
    cmsg = repo.head.object.message.strip()
    logger.log('gitCommitSha= %s'%csha)
    logger.log('gitCommitTime= %s'%ctime)
    logger.log('gitCommitMsg= %s'%cmsg)

    train(args)

def train(args):
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

        ## test
        neps = 1
        paths = []
        print("*** testing ***************************************************")
        for ep_idx in range(neps):
            path = run_one_episode(env, pi, obfilter, render=False)
            paths.append(path)

        logger.record_tabular("TestingNEp", len(paths))
        logger.record_tabular("TestingEpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("TestingEpLenMean", np.mean([path["length"] for path in paths]))
        logger.dump_tabular()

        with open(os.path.join(xprmt_dir,'obfilter.pkl'), 'wb') as f: pickle.dump(obfilter, f)
        env.close()

def test():
    with tf.Session(config=tf.ConfigProto()) as sess2:
        saver = tf.train.import_meta_graph( os.path.join(xprmt_dir,'training_acktr_reacher.meta') )
        saver.restore( sess2,tf.train.latest_checkpoint(xprmt_dir) )

        graph = tf.get_default_graph()

        vs = tf.global_variables('pi')
        print(len(vs))
        print(vs)

def run_one_episode(env, policy, obfilter, render=False):
    ob = env.reset()
    ob = obfilter(ob)
    prev_ob = np.float32(np.zeros(ob.shape))

    obs = []
    acs = []
    ac_dists = []
    logps = []
    rewards = []
    done = False
    step_idx = 0

    while (not done) and (step_idx < env.spec.timestep_limit):
        ## get obs
        concat_ob = np.concatenate([ob, prev_ob], -1)
        obs.append(concat_ob)
        prev_ob = np.copy(ob)

        ## get action
        ac, ac_dist, logp = policy.act(concat_ob)
        acs.append(ac)
        ac_dists.append(ac_dist)
        logps.append(logp)

        scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
        scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)

        ## step
        ob, rew, done, info = env.step(scaled_ac)
        ob = obfilter(ob)
        rewards.append(rew)

        ## closure
        if render:
            print('--- step_idx= %i ---' % step_idx)
            print('scaled_ac= %f, %f' % (scaled_ac[0], scaled_ac[1]))
            print('rew(=dist+ctrl)= %f (=%f + %f)' % (rew,info['reward_dist'],info['reward_ctrl']))
            print(str(ob))
            env.render()
            time.sleep(1/60.)

        step_idx += 1

    return {"observation" : np.array(obs), "reward" : np.array(rewards),
            "action" : np.array(acs), "action_dist": np.array(ac_dists),
            "logp" : np.array(logps), "terminated" : done, "length": len(rewards)}

if __name__ == "__main__":
    main()
