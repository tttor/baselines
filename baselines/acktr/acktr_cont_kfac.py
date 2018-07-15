import time

import numpy as np
import tensorflow as tf

import baselines.common as common
from baselines import logger
from baselines.common import tf_util
from baselines.acktr import kfac

def learn(env, policy, vf, rollout, obfilter, gamma, lam,
          batch_size, max_nsteps, desired_kl=0.002, animate=False):
    # init
    inputs, loss, loss_sampled = policy.update_info
    pi_vars = [var for var in tf.trainable_variables() if 'pi' in var.name]
    lr = tf.Variable(initial_value=np.float32(np.array(0.03)), name='stepsize') # why name stepsize? not lr
    optim = kfac.KfacOptimizer(learning_rate=lr, cold_lr=lr*(1-0.9), momentum=0.9, kfac_update=2,
                               epsilon=1e-2, stats_decay=0.99, async=1, cold_iter=1,
                               weight_decay_dict=policy.wd_dict, max_grad_norm=None)
    update_op, q_runner = optim.minimize(loss, loss_sampled, var_list=pi_vars)
    do_update = tf_util.function(inputs, update_op)
    tf_util.initialize()

    # start queue runners
    enqueue_threads = []
    coord = tf.train.Coordinator()
    for qr in [q_runner, vf.q_runner]:
        assert (qr != None)
        enqueue_threads.extend( qr.create_threads(tf.get_default_session(), coord=coord, start=True) )

    # learning
    batch_idx = 0; total_nsteps = 0
    while total_nsteps < max_nsteps:
        logger.log("***** training batch_idx= %i *****"%batch_idx)

        # Collect paths until we have enough timesteps for this batch
        nsteps = 0; paths = []
        while nsteps < batch_size:
            path = rollout(env, policy, obfilter,
                           render=(len(paths)==0 and (batch_idx % 10 == 0) and animate))
            paths.append(path)
            nsteps += path["length"]

        # Estimate advantage function
        returns = []; advs = []
        for path in paths:
            rew_t = path["reward"]
            return_t = common.discount(rew_t, gamma)
            returns.append(return_t)

            vpred_t = vf.predict(path)
            vpred_t = np.append(vpred_t, 0.0 if path["terminated"] else vpred_t[-1])
            delta_t = (rew_t + gamma*vpred_t[1:]) - vpred_t[:-1]
            adv_t = common.discount(delta_t, gamma * lam)
            advs.append(adv_t)

        # Update value-function network
        vf.fit(paths, returns)

        # Update policy network
        obs = np.concatenate([path["observation"] for path in paths])
        acs = np.concatenate([path["action"] for path in paths])
        advs = np.concatenate(advs)
        standardized_advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        do_update(obs, acs, standardized_advs)

        # Adjust lr
        ac_dists = np.concatenate([path["action_dist"] for path in paths])
        min_lr = np.float32(1e-8)
        max_lr = np.float32(1e0)
        kl = policy.compute_kl(obs, ac_dists)
        if kl > desired_kl * 2:
            logger.log("kl too high")
            tf.assign(lr, tf.maximum(min_lr, lr / 1.5)).eval()
        elif kl < desired_kl / 2:
            logger.log("kl too low")
            tf.assign(lr, tf.minimum(max_lr, lr * 1.5)).eval()
        else:
            logger.log("kl just right!")

        # Close this batch
        total_nsteps += nsteps
        logger.record_tabular("TrainingEpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("TrainingEpLenMean", np.mean([path["length"] for path in paths]))
        logger.record_tabular("TrainingKL", kl)
        logger.record_tabular("TotalNsteps", total_nsteps)
        logger.dump_tabular()
        batch_idx += 1

    coord.request_stop()
    coord.join(enqueue_threads) # Wait for all the threads to terminate.

def run_one_episode(env, policy, obfilter, render=False):
    ob = env.reset()
    ob = obfilter(ob)
    prev_ob = np.float32(np.zeros(ob.shape))
    obs = []; acs = []; ac_dists = []; rewards = [] #; logps = []
    done = False; step_idx = 0; reaching_step_len = env.spec.timestep_limit

    while (not done) and (step_idx < env.spec.timestep_limit):
        ## get obs
        concat_ob = np.concatenate([ob, prev_ob], -1)
        obs.append(concat_ob)
        prev_ob = np.copy(ob)

        ## get action
        ac, ac_dist, logp = policy.act(concat_ob)
        acs.append(ac)
        ac_dists.append(ac_dist)
        # logps.append(logp)

        scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
        scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)

        ## step
        ob, rew, done, info = env.step(scaled_ac)
        ob = obfilter(ob)
        rewards.append(rew)

        # print('=====================')
        # print('step_idx=', step_idx)
        # print('concat_ob=', concat_ob)
        # print('ac=', ac)
        # print('scaled_ac=', scaled_ac)
        # print('ac_dist=', ac_dist)
        # print('logp=', logp)
        # print('rew=', rew)
        # if step_idx==15:
        #     np.savetxt("foo.csv", np.asarray(acs), delimiter=",")
        #     exit()

        if np.isclose(info['reward_dist'], 0.0, atol=0.01):
            reaching_step_len = step_idx + 1 # +1 as index begins at 0

        ## closure
        if render:
            # print('--- step_idx= %i ---' % step_idx)
            # print('done= '+str(done))
            # print('scaled_ac= %f, %f' % (scaled_ac[0], scaled_ac[1]))
            # print('rew(=dist+ctrl)= %f (=%f + %f)' % (rew,info['reward_dist'],info['reward_ctrl']))
            # print(str(ob))
            env.render()
            time.sleep(1/60.)
        step_idx += 1

    return {"observation" : np.array(obs), "reward" : np.array(rewards),
            "action" : np.array(acs), "action_dist": np.array(ac_dists),
            "reaching_step_len": reaching_step_len, # "logp" : np.array(logps),
            "terminated" : done, "length": len(rewards)}
