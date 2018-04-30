# A2C

- Original paper: https://arxiv.org/abs/1602.01783
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.a2c.run_atari`
  runs the algorithm for 40M frames = 10M timesteps on an Atari game.
  See help (`-h`) for more options.

## setup
* `sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev`
  * work if using Python 3.5.2
  * does not work if using Python 3.6.x; problem in installing mpi4py
* `pip3 install opencv-python`
  * https://stackoverflow.com/questions/43019951/after-install-ros-kinetic-cannot-import-opencv
  * https://github.com/ros-perception/vision_opencv/issues/196

## abbreviation for variable names
* pd: probability distribution
* neglogpac: neg log probability action
* td_map: training data map
* for observation shape
  * nh: number of rows (h: height)
  * nw: number of column (w: width)
  * nc: number of channels
* `mb_foo`: mini batch of foo
* adv: advantge
* vf: value function
* lr: learning rate

## fact
* term "model" includes "env"
  * Runner::run() uses both model.step() and env.step()
* `nbatch = nenvs*nsteps`
* nsteps in policy() is only used in LSTM policy
* need to switch to ppo1 for multi CPU
* `nupdates = total_timesteps//nbatch+1`
  * learning is done in batch, `batchsize = nenvs*nsteps`

## question
* nsteps  vs total_timesteps
  * alwyas used the default `nsteps= 5`, why?
  * the target for learning is not nepisodes but total_timesteps?
  * nsteps here refers to nsteps per batch

* python -m baselines.a2c.run_atari does use all cpus
  * is it from the tensorflow only?
  * where is the parallel actor made/called?
    * something todo with `nenvs`, which is set to 16
    * there may be no need to paralle the env.step() becuase it is cheap

* in model class, why passing:
  * reuse=False, in step_model, but
  * reuse=True, in train_model
  * should not both be reuse=True, because step should also reuse variables in the net?

* `loss = pg_loss - entropy*ent_coef + vf_loss*vf_coef`
  * why minus entropy
  * why plus vf_loss

* in /home/tor/ws-fork/baselines@tttor/baselines/a2c/utils.py
  * fc().... stands for? forwad prop? forward pass? fully connected?

* where is return calculated?
  * related to `discount_with_dones()`
  * seems this is general reward formulation, not adjusted to pong
  * `r = reward + gamma*r*(1.-done)`
    * done==True==1 then it is terminal state
```
if dones[-1] == False:
  rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
else:
  rewards = discount_with_dones(rewards, dones, self.gamma)

```

* in Runner::run(): `return mb_obs, mb_states...`, obs vs states

* model.step() vs env.step()
  * model.step(): step fwd (fwd propagate) the policy
    * output: actions, values, states
  * env.step(): step the environment
    * output: obs, rewards, dones

* why should we train nets in a batch?
  * ans: https://datascience.stackexchange.com/questions/16807/why-mini-batch-size-is-better-than-one-single-batch-with-all-training-data

## misc

