# A2C

- Original paper: https://arxiv.org/abs/1602.01783
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.a2c.run_atari`
  runs the algorithm for 40M frames = 10M timesteps on an Atari game.
  See help (`-h`) for more options.

## setup
* python 3.5.2
  * using python 3.6x causes problem in installing mpi4py
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

## fact
* term "model" includes "env"
  * Runner::run() uses both model.step() and env.step()
* `nbatch = nenvs*nsteps`
* nsteps in policy() is only used in LSTM policy
* need to switch to ppo1 for multi CPU
* `nupdates = total_timesteps//nbatch+1`

## question
* nsteps  vs total_timesteps
  * alwyas used the default `nsteps= 5`

* python -m baselines.a2c.run_atari does use all cpus
  * is it from the tensorflow only?
  * where is the parallel actor made/called?
    * something todo with `nenvs`, which is set to 16

* in model class, why passing:
  * reuse=False, in step_model, but
  * reuse=True, in train_model
  * should not both be reuse=True, because step should also reuse variables in the net?

* `loss = pg_loss - entropy*ent_coef + vf_loss*vf_coef`
  * why minus entropy
  * why plus vf_loss

* in /home/tor/ws-fork/baselines@tttor/baselines/a2c/utils.py
  * fc().... stands for? forwad prop? forward pass?
