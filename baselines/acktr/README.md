# ACKTR

* Original paper: https://arxiv.org/abs/1708.05144
* Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
* summary: https://github.com/tttor/rl-foundation/blob/master/method/actor-critic/acktr_wu_2017.md
* run: `(baseline) tor@l7480:~/ws/baselines$ python -m baselines.acktr.run_mujoco -h`

## abbreviation
* com: center of mass
* ev: explained variance, see `baselines/common/math_util.py`
* kl_div: Kullback-Leibler divergence
* lam: lambda
* ob_no: observation with dimension 'n x o'?
* surr: surrogate
* qr: queue runner
* wd_dict: weight data dictionary

## acktr facts
* policy network: Gaussian MLP
* value network: MLP (fully connected, dense)
  * https://www.tensorflow.org/api_docs/python/tf/nn/elu
* is this learning under various init state (jpos, target pose?
  * ans: yes, reset() is called at every rollout()
  * reacher: to always have the same initial joint pos and target pose?
    * diff seed make diff target pose,
      ~~but still not init jointpos~~ and init jointpos
    * however, init jointpos random in only from range [-0.1, +0.1) rad,
      which is small, ~5.72 deg
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

## question
* global seed?
  * seed passed to `make_mujoco_env(args.env, args.seed)`
    indeed control the randomness of environment
    * plus make_mujoco_env() calls `set_global_seeds()`
    * but this does not result in the same final result, why not?
  * seed=0 does not mean using time as seed
    * https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
    * numpy.random.seed() causes numpy to set the seed to a random number obtained from /dev/urandom

## env setup (Python 3.5.2, 3.6.5)
* ubuntu setup
```
sudo apt-get install python3.6-dev
sudo apt-get install libosmesa6-dev

sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
sudo chmod +x /usr/local/bin/patchelf

module load mpi/openmpi-x86_64 # To load mpi in goliath cluter
```

* python setup
```
cd <baseline>
pip install -e .

cd <gym>
pip install -e '.[mujoco]'

pip install gitpython
pip install opencv-python
```

