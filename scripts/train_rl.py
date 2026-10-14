"""Train one of the four released methods on continuous-action CartPole."""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gymnasium as gym
import jax
import numpy as np
from safe_env.register import register, ENV_ID
from relax.buffer import TreeBuffer
from relax.trainer.off_policy import OffPolicyTrainer
from scripts.common import METHODS, build_algorithm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alg', choices=METHODS, default='sac-rfsi-dr-s')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--hidden_num', type=int, default=2)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--start_step', type=int, default=10000)
    p.add_argument('--total_step', type=int, default=600000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--buffer_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--certificate_lr', type=float, default=3e-4, help='FRI methods only.')
    p.add_argument('--multiplier_lr', type=float, default=None,
                   help='Default: 1e-4 for FRI methods, 3e-4 for SAC-Lag.')
    p.add_argument('--feasible_threshold', type=float, default=0.1, help='FRI methods only.')
    p.add_argument('--infeasible_threshold', type=float, default=0.9, help='FRI methods only.')
    p.add_argument('--save_every', type=int, default=10000)
    p.add_argument('--log_every', type=int, default=1000)
    p.add_argument('--log_dir', type=Path, default=None)
    args = p.parse_args()
    if args.multiplier_lr is None:
        args.multiplier_lr = 3e-4 if args.alg == 'sac-lag-d' else 1e-4
    if not (1 <= args.start_step <= args.total_step):
        p.error('Require 1 <= start_step <= total_step.')
    if min(args.hidden_num, args.hidden_dim, args.batch_size, args.save_every, args.log_every) <= 0:
        p.error('Sizes and intervals must be positive.')
    if args.buffer_size is not None and args.buffer_size <= 0:
        p.error('buffer_size must be positive.')
    log_dir = args.log_dir or ROOT / 'logs' / ENV_ID / f'{args.alg}_{datetime.now():%Y%m%d_%H%M%S_%f}_s{args.seed}'
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config['log_dir'] = str(log_dir)
    (log_dir / 'config.json').write_text(json.dumps(config, indent=2))
    rng = np.random.default_rng(args.seed)
    env_seed, buffer_seed, net_seed, train_seed = map(int, rng.integers(0, 2**32 - 1, 4))
    register()
    env = gym.make(ENV_ID, seed=env_seed, output_dir=str(log_dir))
    env.action_space.seed(env_seed)
    try:
        algorithm = build_algorithm(args, env, net_seed)
        buffer = TreeBuffer.from_experience(env.observation_space.shape[0], env.action_space.shape[0],
                                           size=args.buffer_size or args.total_step, seed=buffer_seed)
        trainer = OffPolicyTrainer(env, algorithm, buffer, log_dir, batch_size=args.batch_size,
                                   start_step=args.start_step, total_step=args.total_step,
                                   save_every=args.save_every, log_every=args.log_every)
        print(f'Algorithm: {args.alg}; backend: {jax.default_backend()}; output: {log_dir}', flush=True)
        summary = trainer.train(jax.random.PRNGKey(train_seed))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
