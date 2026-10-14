"""Verify four methods in a copied release directory, outside the source tree."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['sac-rfsi-dr-s', 'sac-fsi-dr', 'sac-fsi-a', 'sac-lag-d']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--alg', choices=METHODS, nargs='+', default=METHODS,
                   help='Methods to check; defaults to all four.')
    args = p.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix='drsrl-isolated-') as tmp:
        isolated = Path(tmp) / 'release'
        shutil.copytree(ROOT, isolated, ignore=shutil.ignore_patterns('__pycache__', 'logs', '*.egg-info', '.venv', 'build', 'dist'))
        env = os.environ.copy()
        env['RELAX_USE_CUDA'] = '0'
        env['JAX_PLATFORM_NAME'] = 'cpu'
        env['PYTHONPATH'] = ''
        check_code = r"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import gymnasium as gym
import numpy as np
import relax, safe_env, safe_control_gym
from safe_env.register import register, ENV_ID
root = Path(sys.argv[1]).resolve()
for module in [relax, safe_env, safe_control_gym]:
    assert Path(module.__file__).resolve().is_relative_to(root)
register()
# Exercise device-to-host insertion and wraparound in the replay buffer.
import jax
from relax.buffer import TreeBuffer
from relax.utils.experience import Experience
buffer = TreeBuffer.from_experience(4, 1, size=2, seed=0)
example = Experience.create_example(4, 1)._replace(obs=np.ones(4, dtype=np.float32))
buffer.add(jax.device_put(example), from_jax=True)
np.testing.assert_array_equal(buffer.sample(1).obs, np.ones((1, 4)))
batch = Experience.create_example(4, 1, 2)._replace(obs=np.full((2, 4), 2, dtype=np.float32))
buffer.add_batch(jax.device_put(batch), from_jax=True)
np.testing.assert_array_equal(buffer.sample(4, to_jax=True).obs, np.full((4, 4), 2))
with tempfile.TemporaryDirectory() as output:
    env = gym.make(ENV_ID, output_dir=output)
    try:
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)
    finally:
        env.close()
    env = gym.make(ENV_ID, output_dir=output, randomized_init=False,
                   init_state=np.zeros(4))
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(np.zeros(1))
        assert terminated and not truncated
        assert info['feasible'] and not info['infeasible'] and info['cost'] == 0
        env.close()
        env = gym.make(ENV_ID, output_dir=output, randomized_init=False,
                       init_state=np.array([0.2, 0, 0, 0]))
        env.reset(seed=0)
        for step in range(1, 251):
            obs, reward, terminated, truncated, info = env.step(np.zeros(1))
            assert not terminated
            assert truncated == (step == 250)
            assert not info['feasible'] and not info['infeasible'] and info['cost'] == 0
    finally:
        env.close()
print('Environment API, seed, horizon, safety labels and local imports: PASS')
"""
        subprocess.run([sys.executable, '-c', check_code, str(isolated)],
                       cwd=tmp, env=env, check=True)
        for method in args.alg:
            run_dir = Path(tmp) / method
            command = [sys.executable, str(isolated / 'scripts/train_rl.py'), '--alg', method,
                       '--total_step', '260', '--start_step', '252', '--batch_size', '8',
                       '--hidden_dim', '16', '--log_every', '260', '--save_every', '260',
                       '--log_dir', str(run_dir)]
            subprocess.run(command, cwd=tmp, env=env, check=True)
            # Check checkpoint restoration without adding an evaluation protocol.
            load_code = r"""
import json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import gymnasium as gym
import numpy as np
from scripts.common import build_algorithm
from safe_env.register import register, ENV_ID
run_dir = Path(sys.argv[2])
config = SimpleNamespace(**json.loads((run_dir / 'config.json').read_text()))
register()
with tempfile.TemporaryDirectory() as output:
    env = gym.make(ENV_ID, output_dir=output)
    try:
        algorithm = build_algorithm(config, env, 0)
        algorithm.load(run_dir / 'state.pkl')
        assert int(algorithm.state.step) == 9
        obs, _ = env.reset(seed=100)
        action = algorithm.get_deterministic_action(obs)
        assert action.shape == env.action_space.shape and np.all(np.isfinite(action))
    finally:
        env.close()
print('Checkpoint restore: PASS')
"""
            subprocess.run([sys.executable, '-c', load_code, str(isolated), str(run_dir)],
                           cwd=tmp, env=env, check=True)
            summary = json.loads((run_dir / 'summary.json').read_text())
            assert summary['updates'] == 9
            assert (run_dir / 'state_260.pkl').exists()
            assert summary['episodes'] >= 1
            results.append(dict(method=method, passed=True, steps=summary['steps'],
                                updates=summary['updates'], episodes=summary['episodes'],
                                checkpoint_restored=True))
            print(f'PASS: {method}', flush=True)
    report = dict(python=sys.version, isolated_copy=True, backend='cpu', environment_checks=True, replay_buffer_checks=True, results=results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
