# ACKTR

* https://github.com/tttor/rl-foundation/blob/master/method/actor-critic/acktr_wu_2017.md
* run:
```
(baseline) tor@l7480:~/ws/baselines$ python -m baselines.acktr.run_mujoco -h
(baseline) tor@l7480:~/ws/baselines$ python -m baselines.acktr.run_mujoco --mode train  --env Reacher-v2@010 --nsteps 1000 --seed 0 --dir ~/xprmt/xprmt-acktr
```

## acktr facts
* actor/policy network
  * Gaussian MLP
* critic/value network:
  * MLP (fully connected, dense)
  * predict Advantage values, not Q values
  * https://www.tensorflow.org/api_docs/python/tf/nn/elu
* is this learning under various init state (jpos, target pose)?
  * ans: yes, reset() is called at every rollout()
* loss vs loss_sampled?
  * ans: loss = surr
```
surr = - tf.reduce_mean(adv_n * logprob_n)
surr_sampled = - tf.reduce_mean(logprob_n) # Sampled loss of the policy
```
* kfac
  * baselines/acktr/kfac.py
  * https://www.tensorflow.org/api_docs/python/tf/contrib/kfac/optimizer/KfacOptimizer
  * https://arxiv.org/abs/1503.05671
* multi-threading is for optimization (network operations), not for rollout
  * https://www.tensorflow.org/api_docs/python/tf/train/QueueRunner
  * https://www.tensorflow.org/api_docs/python/tf/train/Coordinator
* observation filter is crucial!
  * `y = (x-mean)/std`
    using running estimates of mean,std
* kfac for both actor and critic, see `learn()` and `critic_net`

## question
* global seed?
  * seed passed to `make_mujoco_env(args.env, args.seed)`
    indeed control the randomness of environment
    * plus make_mujoco_env() calls `set_global_seeds()`
    * but this does not result in the same final result, why not?
  * seed=0 does not mean using time as seed
    * https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
    * numpy.random.seed() causes numpy to set the seed to a random number obtained from /dev/urandom

## abbreviation (mostly used in variable naming)
* com: center of mass
* ev: explained variance, see `baselines/common/math_util.py`
* kl_div: Kullback-Leibler divergence
* lam: lambda
* ob_no: observation with dimension 'n x o'?
* surr: surrogate
* qr, q_runner: queue runner
* wd_dict: weight data dictionary

## env setup (Python 3.5.2, 3.6.5)
* ubuntu setup
```
sudo apt-get install python3.6-dev
sudo apt-get install libosmesa6-dev

sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
sudo chmod +x /usr/local/bin/patchelf
```

* python setup
```
module load mpi/openmpi-x86_64 # load mpi in goliath cluster

cd <baselines>
pip install -e .

cd <gym>
pip install -e '.[mujoco]'

pip install gitpython
pip install opencv-python
```

## tested perf
* Reacher-v2@010: nsteps= 500K
```
gitCommitSha= 2382c77b942e1ab61180ef3f3b5e8c1c49cd9f13
gitCommitTime= Sun Jun 10 13:40:28 2018
gitCommitMsg= fix
seed= 0
***** training batch_idx= 199 *****
kl just right!
------------------------------------
| EVAfter           | 0.964        |
| EVBefore          | 0.962        |
| TotalNsteps       | 500000       |
| TrainingEpLenMean | 50           |
| TrainingEpRewMean | -4.87        |
| TrainingKL        | 0.0012952603 |
------------------------------------
***** immediate testing *****
-------------------------------
| TestingEpLenMean | 50       |
| TestingEpRewMean | -5.08    |
| TestingNEps      | 100      |
-------------------------------
```

* Reacher-v2@010: nsteps= 400K
```sh
gitCommitSha= 5224d66fcaff1f7db2a94d60856896840b7fc70d
gitCommitTime= Mon May  7 12:00:31 2018
gitCommitMsg= ob filter init in main
...
********** training batch_idx= 159 ************
kl too high
-----------------------------------
| EVAfter           | 0.938       |
| EVBefore          | 0.928       |
| TrainingEpLenMean | 50          |
| TrainingEpRewMean | -5.96       |
| TrainingKL        | 0.004674497 |
-----------------------------------

```
