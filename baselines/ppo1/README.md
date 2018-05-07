# PPOSGD

* Original paper: https://arxiv.org/abs/1707.06347
* Baselines blog post: https://blog.openai.com/openai-baselines-ppo/
* runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
```mpirun -np 8 python -m baselines.ppo1.run_atari```
* runs the algorithm for 1M frames on a Mujoco environment.
```python -m baselines.ppo1.run_mujoco --env Reacher-v2 --seed 0 --num-timesteps 1000```

- Train mujoco 3d humanoid (with optimal-ish hyperparameters): `mpirun -np 16 python -m baselines.ppo1.run_humanoid --model-path=/path/to/model`
- Render the 3d humanoid: `python -m baselines.ppo1.run_humanoid --play --model-path=/path/to/model`
