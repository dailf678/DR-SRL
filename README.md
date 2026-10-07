# Safe Reinforcement Learning with Dual Robustness via Robust Control Barrier Functions

This repository provides the key implementation for the paper:

**Safe Reinforcement Learning with Dual Robustness via Robust Control Barrier Functions**

The currently released code focuses on the **Cartpole** experiment and contains the main components required to show the proposed safe reinforcement learning framework, including the algorithm implementation, network architecture, training pipeline, and safety-critical Cartpole environment.

> **Code availability.** This repository currently contains the key Cartpole implementation. The implementations for the remaining simulation environments will will be coming soon.

## Repository Structure

```text
.
├── relax/
│   ├── algorithm/
│   │   └── sac_rfsi_DR_s.py
│   ├── network/
│   │   └── sac_rfsi_DR_s.py
│   └── trainer/
│       └── off_policy.py
├── safe_env/
│   └── safe_control_gym/
└── scripts/
    └── train_rl.py
```

The main files are organized as follows.

### Algorithm

```text
relax/algorithm/sac_rfsi_DR_s.py
```

Implements the proposed safe reinforcement learning algorithm, including the main optimization and parameter-update procedures.

### Network Architecture

```text
relax/network/sac_rfsi_DR_s.py
```

Defines the neural-network architectures used by the algorithm.

### Training Pipeline

```text
relax/trainer/off_policy.py
```

Implements the off-policy training loop. The trainer performs:

### Training Entry Point

```text
scripts/train_rl.py
```

Provides the entry point for starting the reinforcement learning training procedure.

### Cartpole Environment

```text
safe_env/safe_control_gym/
```

Contains the safety-critical Cartpole environment used in the released experiment.
