import numpy as np
import tensorflow as tf

import baselines.common as common
from baselines import logger
from baselines.common import tf_util as U

from baselines.acktr import kfac

def learn(env, policy, vf, rollout, obfilter, gamma, lam,
          batch_size, max_nsteps, desired_kl=0.002, animate=False):
    lr = tf.Variable(initial_value=np.float32(np.array(0.03)), name='stepsize') # why name stepsize? not lr
    optim = kfac.KfacOptimizer(learning_rate=lr, cold_lr=lr*(1-0.9), momentum=0.9, kfac_update=2,
                               epsilon=1e-2, stats_decay=0.99, async=1, cold_iter=1,
                               weight_decay_dict=policy.wd_dict, max_grad_norm=None)
    pi_vars = [var for var in tf.trainable_variables() if 'pi' in var.name]
    inputs, loss, loss_sampled = policy.update_info
    update_op, q_runner = optim.minimize(loss, loss_sampled, var_list=pi_vars)
    do_update = U.function(inputs, update_op)
    U.initialize()

    # start queue runners
    enqueue_threads = []
    coord = tf.train.Coordinator()
    for qr in [q_runner, vf.q_runner]:
        assert (qr != None)
        enqueue_threads.extend( qr.create_threads(tf.get_default_session(), coord=coord, start=True) )

    batch_idx = 0; total_nsteps = 0
    while total_nsteps < max_nsteps:
        logger.log("********** training batch_idx= %i ************"%batch_idx)

        # Collect paths until we have enough timesteps for this batch
        nsteps = 0; paths = []
        while nsteps < batch_size:
            path = rollout(env, policy, obfilter,
                           render=(len(paths)==0 and (batch_idx % 10 == 0) and animate))
            paths.append(path)
            n = path["length"]
            nsteps += n

        # Estimate advantage function
        vtargs = []; advs = []
        for path in paths:
            rew_t = path["reward"]
            return_t = common.discount(rew_t, gamma)
            vtargs.append(return_t)

            vpred_t = vf.predict(path)
            vpred_t = np.append(vpred_t, 0.0 if path["terminated"] else vpred_t[-1])
            delta_t = (rew_t + gamma*vpred_t[1:]) - vpred_t[:-1]
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

        # Adjust lr
        min_lr = np.float32(1e-8)
        max_lr = np.float32(1e0)
        kl = policy.compute_kl(ob_no, oldac_dist)
        if kl > desired_kl * 2:
            logger.log("kl too high")
            tf.assign(lr, tf.maximum(min_lr, lr / 1.5)).eval()
        elif kl < desired_kl / 2:
            logger.log("kl too low")
            tf.assign(lr, tf.minimum(max_lr, lr * 1.5)).eval()
        else:
            logger.log("kl just right!")

        # Close this batch
        logger.record_tabular("TrainingEpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("TrainingEpLenMean", np.mean([path["length"] for path in paths]))
        logger.record_tabular("TrainingKL", kl)
        logger.dump_tabular()
        total_nsteps += nsteps
        batch_idx += 1

    coord.request_stop()
    coord.join(enqueue_threads)
