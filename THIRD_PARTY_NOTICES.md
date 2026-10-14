# Source and attribution

- Main algorithm and network: `dailf678/DR-SRL`.
- Baseline algorithms/networks and shared `relax` infrastructure: copied from the local Feasible-Region-Iteration research project. The upstream project is https://github.com/yangyujie-jack/Feasible-Region-Iteration. The local checkout did not contain a root license file. No new license is assigned to these files by this packaging task.
- `safe_control_gym`: minimal subset of the local modified safe-control-gym checkout. The upstream project is https://github.com/utiasDSL/safe-control-gym. Its MIT copyright/license is preserved in `safe_control_gym/LICENSE`. The CartPole file also retains its original source attribution comments. All copied implementation files are unchanged except the two extracted NumPy utilities and minimal package initializers. The Gymnasium adapter is in `safe_env/`.
- Python packages listed in `requirements.txt`, including PyBullet and its `pybullet_data`, are installed dependencies, not vendored copies.

The distribution excludes original training logs, checkpoints, figures, maps, archives, unrelated algorithms and non-CartPole environments.
