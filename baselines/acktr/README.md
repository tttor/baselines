# ACKTR

- Original paper: https://arxiv.org/abs/1708.05144
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
  * `python -m baselines.acktr.run_atari`

## env setup (Python 3.5.2, 3.6.5)
* sudo apt-get install python3.6-dev
* sudo apt-get install libosmesa6-dev
* sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
* sudo chmod +x /usr/local/bin/patchelf
* ref
  * https://github.com/openai/mujoco-py/issues/47
  * https://github.com/openai/mujoco-py/issues/96
    * https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory
