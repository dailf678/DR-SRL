# DR-SRL — CartPole release

Runnable CartPole code for **Safe Reinforcement Learning with Dual Robustness via Robust Control Barrier Functions**.
This release contains only four methods and their shared training code:

| Command-line name | Paper method | Algorithm / network file | Source |
| --- | --- | --- | --- |
| `sac-rfsi-dr-s` | FRI-Lipschitz (local bound) | `sac_rfsi_DR_s.py` | Published DR-SRL implementation |
| `sac-fsi-dr` | FRI-Minimax | `sac_fsi_DR.py` | Local baseline |
| `sac-fsi-a` | FRI-Nominal | `sac_fsi_A.py` | Local baseline |
| `sac-lag-d` | SAC-Lag | `sac_lag_d.py` | Local baseline |

The package includes the four CartPole methods, their shared training pipeline, and the environment source needed to run them. FRI-Minimax and FRI-Nominal use adversarial action amplitudes of 0.2.

## Installation

Linux, Python 3.10 or 3.11. CPU training requires no CUDA or GPU. From this directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Alternatively, run `conda env create -f environment.yml` here, then `conda activate drsrl-cartpole` and `python -m pip install --no-deps -e .`.
The requirements pin a compatible CPU JAX stack. `requirements-lock.txt` additionally records all installed package versions from the clean Python 3.10 validation environment; use `python -m pip install -r requirements-lock.txt` for that complete set. They intentionally do not copy the existing machine's CUDA packages. All project source files and the CartPole URDF are included; no sibling checkout, downloaded training data, or pretrained model is needed.

## Validate the release

```bash
python scripts/smoke_test.py --output validation.json
```

This copies the release into a temporary directory, clears `PYTHONPATH`, checks environment seeding, goal termination, safety labels and the 250-step time limit, checks replay-buffer device-to-host insertion and wraparound, and tests all four methods on CPU. Each run collects 260 steps, performs nine actual gradient updates, crosses an episode boundary, writes checkpoints, reloads the final checkpoint, and checks that the restored policy produces a finite action with the correct shape. Small smoke-test networks are for testing functionality; this does not establish convergence or reproduce paper scores.

## Train

```bash
python scripts/train_rl.py --alg sac-rfsi-dr-s --seed 1
python scripts/train_rl.py --alg sac-fsi-dr --seed 1
python scripts/train_rl.py --alg sac-fsi-a --seed 1
python scripts/train_rl.py --alg sac-lag-d --seed 1
```

Or run all methods sequentially:

```bash
bash scripts/train_all.sh --seed 1
```

Each invocation trains once. Run separate seeds explicitly, for example `bash scripts/train_all.sh --seed 2`. All options are listed by `python scripts/train_rl.py --help`. You can invoke the script by its absolute path from another working directory.

Defaults: 600,000 environment steps, 10,000 random exploration steps, two hidden layers of width 128, batch size 256, learning rate `3e-4`, certificate learning rate `3e-4`, multiplier learning rate `1e-4`, feasible/infeasible thresholds `0.1`/`0.9`. SAC-Lag defaults to multiplier learning rate `3e-4`; `--multiplier_lr` works for all four methods, with defaults selected by method. Certificate learning rate and feasibility thresholds apply to the FRI methods.

Outputs go into a new `logs/DRSRL-CartPole-v0/<method>_<timestamp>_s<seed>/` directory:

- `config.json`: command-line configuration used for this run.
- TensorBoard events: episode returns, costs, lengths, and update metrics.
- `state_<step>.pkl`: periodic checkpoints, every 10,000 steps by default.
- `state.pkl`: final checkpoint, including optimizer state.
- `summary.json`: step/update counts and final metrics.

For a custom destination use `--log_dir /path/to/new/run`; the destination must not already exist. TensorBoard:

```bash
tensorboard --logdir logs
```

## Disturbances and method interpretation

All four methods retain two learned adversaries for collecting transitions under a comparable disturbance environment: each step selects the safety or task adversary with probability 0.5, and each disturbance has magnitude at most 0.2 in the one-dimensional CartPole action space. FRI-Nominal learns and samples both adversaries with amplitude 0.2; its task-policy loss evaluates nominal reward and safety terms without adversarial actions or a Lipschitz margin.

FRI-Lipschitz and FRI-Minimax also use the safety adversary in certificate synthesis; FRI-Nominal retains nominal certificate synthesis. In FRI-Lipschitz task-policy optimization, the reward term uses the task adversary, while the safety constraint uses the adversary-independent Lipschitz margin. The safety adversary is not used in that policy safety term. SAC-Lag's lack of explicit disturbance robustness refers to its safety constraint, not to the absence of disturbed training samples.

This release supplies training and checkpoint validation. It does not supply an experimental evaluation protocol. The CartPole safety bounds match the paper; full agreement of hyperparameters, the fixed input-matrix bound, and supplementary implementation details has not been verified against Table S-I.

## Environment and compatibility changes

`DRSRL-CartPole-v0` is the original continuous-action **PyBullet** environment, rather than Gym's built-in discrete-action CartPole. The state order is `[x, x_dot, theta, theta_dot]`; the normalized nominal action space is `[-1, 1]`. The selected network code retains its additive adversarial action logic. Control and physics frequency are 50 Hz; episodes last at most five seconds (250 steps). Initial `x` is sampled from `[-0.5, 0.5]` and `theta` from `[-0.1, 0.1]`. Safety constraints are `|theta| <= 0.2` and `|theta_dot| <= 0.2`.

The `safe_control_gym/` directory vendors only the local CartPole dependency chain and its URDF. Package initializers no longer import unrelated controllers, quadrotors or Torch. Two NumPy utility functions are extracted unchanged from their original modules. The environment dynamics, rewards, bounds and constraints retain the local implementation.

The environment wrapper returns Gymnasium's `(obs, info)` reset result and `(obs, reward, terminated, truncated, info)` step result. Feasible/infeasible labels use the pre-step state, while constraint costs use the post-step state. The trainer resets on either episode-ending flag and stores `done = terminated or truncated` in replay. Seeds are applied to the environment and its action space. A final checkpoint is written even for short runs.

## Attribution

Shared RL infrastructure is derived from [Feasible-Region-Iteration](https://github.com/yangyujie-jack/Feasible-Region-Iteration). The main method comes from [DR-SRL](https://github.com/dailf678/DR-SRL). The included CartPole environment is a subset of the locally modified [safe-control-gym](https://github.com/utiasDSL/safe-control-gym); its MIT license is retained in `safe_control_gym/LICENSE`. See `THIRD_PARTY_NOTICES.md` for details. This directory does not assign a new license to the authors' method code.
