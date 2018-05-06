# ACKTR

- Original paper: https://arxiv.org/abs/1708.05144
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- runs the algorithm for 40M frames = 10M timesteps on an Atari game.
  See help (`-h`) for more options.
  * `python -m baselines.acktr.run_atari`
- runs in mujoco env
  * `python -m baselines.acktr.run_mujoco --env Reacher-v2 --seed 0 --num-timesteps 1000`

## env setup (Python 3.5.2, 3.6.5)
* sudo apt-get install python3.6-dev
* sudo apt-get install libosmesa6-dev
* add patchef
```
sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
sudo chmod +x /usr/local/bin/patchelf
```
* ref
  * https://github.com/openai/mujoco-py/issues/47
  * https://github.com/openai/mujoco-py/issues/96
  * https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory

## abbreviation
* kl_div: Kullback-Leibler divergence
* wd_dict: weight data dictionary
* ob_no: ?
* com: center of mass

## fact
* policy network: Gaussian MLP
* value network: ?
* standard reacher has frameskip(timestep skip)= 2
* reacher ob.shape= 11
  * 2: sin(theta) of 2 joints
  * 2: cos(theta) of 2 joints
  * 2: qpos of target, x and y slide joints
  * 2: qvel of 2 joints
  * 3: distance between fingertip and target (3D cartesian)
* reacher, where is env.spec.timestep_limit=50 per episode defined?
  * ans: /home/tor/ws-fork/gym@tttor/gym/envs/__init__.py
* is this learning under various init state (jpos, target pose?
  * ans: yes, reset() is called at every rollout()
  * reacher: to always have the same initial joint pos and target pose?
    * diff seed make diff target pose,
      ~~but still not init jointpos~~ and init jointpos
    * however, init jointpos random in only from range [-0.1, +0.1) rad,
      which is small, ~5.72 deg

## question
* loss vs loss_sampled?
* global seed?
  * seed passed to `make_mujoco_env(args.env, args.seed)`
    indeed control the randomness of environment
    * plus make_mujoco_env() calls `set_global_seeds()`
    * but this does not result in the same final result, why not?
  * seed=0 does not mean using time as seed
    * https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
    * numpy.random.seed() causes numpy to set the seed to a random number obtained from /dev/urandom
* who makes it run parallel using cpu?
* for Reacher-v2:
  why set: reward_threshold=-3.75? instead of `=0`?
  is it sort of tolerance?



