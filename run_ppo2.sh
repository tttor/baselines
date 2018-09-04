#!/bin/bash
if [ "$#" -ne 1 ]; then
  echo "USAGE:"
  echo "bash run.sh <env_id>"
  exit 1
fi
env_id=${1}

source ~/venv/baselines/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro150/bin

python -m baselines.run \
--env=$env_id \
--seed=12 \
--alg=ppo2 \
--num_timesteps=1e6 \
--network=mlp \
--num_env=1 \
--reward_scale=1.0
