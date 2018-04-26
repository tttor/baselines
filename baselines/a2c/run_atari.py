#!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from baselines.a2c import a2c
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    logger.configure()

    policy=args.policy
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    else:
        return

    env = VecFrameStack(make_atari_env(env_id=args.env, num_env=16, seed=args.seed), nstack=4)

    a2c.learn(policy_fn, env, args.seed, nsteps=1,
              total_timesteps=int(args.num_timesteps * 1.1), lrschedule=args.lrschedule)

    env.close()

if __name__ == '__main__':
    main()
