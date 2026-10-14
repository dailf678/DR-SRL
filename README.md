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
