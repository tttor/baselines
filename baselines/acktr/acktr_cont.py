import numpy as np
import tensorflow as tf
from baselines import logger
import baselines.common as common
from baselines.common import tf_util as U
from baselines.acktr import kfac
from baselines.acktr.filters import ZFilter

def rollout(env, policy, max_pathlength, render=False, obfilter=None):
    """
    Simulate the env and policy for max_pathlength steps
    """
    ob = env.reset()
    prev_ob = np.float32(np.zeros(ob.shape))
    if obfilter: ob = obfilter(ob)
    terminated = False
    obs = []
    acs = []
    ac_dists = []
    logps = []
    rewards = []

    for _ in range(max_pathlength):
        if render: env.render()

        ## get obs
        state = np.concatenate([ob, prev_ob], -1)
        obs.append(state)
        prev_ob = np.copy(ob)

        ## get action
        ac, ac_dist, logp = policy.act(state)
        acs.append(ac)
        ac_dists.append(ac_dist)
        logps.append(logp)

        scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
        scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)

        ## step
        ob, rew, done, _ = env.step(scaled_ac)

        rewards.append(rew)
        if obfilter:
            ob = obfilter(ob)
        if done:
            terminated = True
            break

    return {"observation" : np.array(obs), "terminated" : terminated,
            "reward" : np.array(rewards), "action" : np.array(acs),
            "action_dist": np.array(ac_dists), "logp" : np.array(logps)}

def learn(env, policy, vf, gamma, lam, timesteps_per_batch, num_timesteps,
          animate=False, callback=None, desired_kl=0.002):

    obfilter = ZFilter(env.observation_space.shape)
    max_pathlength = env.spec.timestep_limit

    lr = tf.Variable(initial_value=np.float32(np.array(0.03)), name='stepsize') # why name stepsize?
    inputs, loss, loss_sampled = policy.update_info
    optim = kfac.KfacOptimizer(learning_rate=lr, cold_lr=lr*(1-0.9), momentum=0.9, kfac_update=2,\
                                epsilon=1e-2, stats_decay=0.99, async=1, cold_iter=1,
                                weight_decay_dict=policy.wd_dict, max_grad_norm=None)
    pi_var_list = []
    for var in tf.trainable_variables():
        if "pi" in var.name:
            pi_var_list.append(var)

    update_op, q_runner = optim.minimize(loss, loss_sampled, var_list=pi_var_list)
    do_update = U.function(inputs, update_op)
    U.initialize()

    # start queue runners
    enqueue_threads = []
    coord = tf.train.Coordinator()
    for qr in [q_runner, vf.q_runner]:
        assert (qr != None)
        enqueue_threads.extend( qr.create_threads(tf.get_default_session(), coord=coord, start=True) )

    iter_idx = 0
    timesteps_so_far = 0
    while True:
        if timesteps_so_far > num_timesteps:
            break
        logger.log("********** Learning Iteration % i ************"%iter_idx)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            path = rollout(env, policy, max_pathlength,
                           render=(len(paths)==0 and (iter_idx % 10 == 0) and animate),
                           obfilter=obfilter)

            paths.append(path)
            n = _pathlength(path)
            timesteps_this_batch += n
            timesteps_so_far += n
            if timesteps_this_batch > timesteps_per_batch:
                break

        # Estimate advantage function
        vtargs = []
        advs = []
        for path in paths:
            rew_t = path["reward"]
            return_t = common.discount(rew_t, gamma)
            vtargs.append(return_t)
            vpred_t = vf.predict(path)
            vpred_t = np.append(vpred_t, 0.0 if path["terminated"] else vpred_t[-1])
            delta_t = rew_t + gamma*vpred_t[1:] - vpred_t[:-1]
            adv_t = common.discount(delta_t, gamma * lam)
            advs.append(adv_t)

        # Update value function
        vf.fit(paths, vtargs)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        action_na = np.concatenate([path["action"] for path in paths])
        oldac_dist = np.concatenate([path["action_dist"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        # Policy update
        do_update(ob_no, action_na, standardized_adv_n)

        min_lr = np.float32(1e-8)
        max_lr = np.float32(1e0)

        # Adjust lr
        kl = policy.compute_kl(ob_no, oldac_dist)
        if kl > desired_kl * 2:
            logger.log("kl too high")
            tf.assign(lr, tf.maximum(min_lr, lr / 1.5)).eval()
        elif kl < desired_kl / 2:
            logger.log("kl too low")
            tf.assign(lr, tf.minimum(max_lr, lr * 1.5)).eval()
        else:
            logger.log("kl just right!")

        logger.record_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("EpRewSEM", np.std([path["reward"].sum()/np.sqrt(len(paths)) for path in paths]))
        logger.record_tabular("EpLenMean", np.mean([_pathlength(path) for path in paths]))
        logger.record_tabular("KL", kl)

        if callback: callback()
        logger.dump_tabular()
        iter_idx += 1

    coord.request_stop()
    coord.join(enqueue_threads)

def _pathlength(path):
    return path["reward"].shape[0]# Loss function that we'll differentiate to get the policy gradient
