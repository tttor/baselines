# ACKTR

* https://github.com/tttor/rl-foundation/blob/master/method/actor-critic/acktr_wu_2017.md
* run:
```
(baseline) tor@l7480:~/ws/baselines$ python -m baselines.acktr.run_mujoco -h
(baseline) tor@l7480:~/ws/baselines$ python -m baselines.acktr.run_mujoco --mode train  --env Reacher-v2@010 --nsteps 1000 --seed 0 --dir ~/xprmt/openai-baselines-acktr
```

# arch, hyperparam
## actor, policy network
* named: gaussian mlp
* activ fn
  * tanh - tanh - linear
* weight init
  * custom: normc_initializer;
    /home/tor/ws/baselines/baselines/common/tf_util.py
  * NOTE: output weight, std=0.1
* bias init
  * zeroed!

## critic, valuefn network
* input_dim= 28
  * observation, dim= `n x 22`
  * act_dist, dim= `n x 4`
  * ? al?, dim= `n x 1`
  * bias: ones, dim = `n x 1`
* activ-fn
  * elu - elu - dense
* n unit
  * 64 - 64 - 1
* weight init
  * custom: normc_initializer;
    /home/tor/ws/baselines/baselines/common/tf_util.py
* bias init
  * zeroed!
* loss
  * not really MSE:
  `loss = tf.reduce_mean(tf.square(vpred_n - vtarg_n)) + tf.add_n(wd_loss)`
* update
  * nepoch=25 per batch
  `for _ in range(25): self.do_update(X, y)`

## agent param
* gamma=0.99, lam=0.97,
* psi

## env
* use Monitor from /home/tor/ws/baselines/baselines/bench/monitor.py

# question
* how policy update work?
  how these args are used in `do_update(...)`:
  `(obs, acs, standardized_advs)` ?
* lambda? is it related to generalized advantage estimation (GAE)?
```py
def learn():
  adv_t = common.discount(delta_t, gamma * lam)
```
* break down input X for vf.predict(), dim= `n x 28`
  * observation, dim= `n x 22`
  * act_dist, dim= `n x 4`
  * ? al?, dim= `n x 1`
  * bias: ones, dim = `n x 1`
```py
def _preproc(self, path):
    l = path["reward"].shape[0]
    al = np.arange(l).reshape(-1,1)/10.0
    act = path["action_dist"].astype('float32')
    X = np.concatenate([path['observation'], act, al, np.ones((l, 1))], axis=1)
    return X

def predict(self, path):
    return self._predict(self._preproc(path))
```
* global seed?
  * seed passed to `make_mujoco_env(args.env, args.seed)`
    indeed control the randomness of environment
    * plus make_mujoco_env() calls `set_global_seeds()`
    * but this does not result in the same final result, why not?
  * seed=0 does NOT mean using time as seed
    * https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
    * numpy.random.seed() causes numpy to set the seed to a random number obtained from /dev/urandom

# abbreviation (mostly used in variable naming)
* com: center of mass
* ev: explained variance, see `baselines/common/math_util.py`
* kl_div: Kullback-Leibler divergence
* lam: lambda
* ob_no: observation with dimension 'n x o'?
* surr: surrogate
* qr, q_runner: queue runner
* wd_dict: weight data dictionary

# env setup (Python 3.5.2, 3.6.5)
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

# tested perf
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
