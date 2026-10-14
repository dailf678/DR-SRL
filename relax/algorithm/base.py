"""Common CPU/GPU interface for the four released algorithms."""
import pickle
from pathlib import Path

import jax
import numpy as np


class Algorithm:
    def _implement_common_behavior(self, stateless_update, stateless_get_action,
                                   stateless_get_deterministic_action):
        self._update = jax.jit(stateless_update)
        self._get_action = jax.jit(stateless_get_action)
        self._get_deterministic_action = jax.jit(stateless_get_deterministic_action)

    def update(self, key, data):
        self.state, info = self._update(key, self.state, jax.device_put(data))
        return {k: float(v) for k, v in info.items()}

    def get_action(self, key, obs):
        p = self.state.params
        return np.asarray(self._get_action(key, p.policy, p.safe_adversary_policy,
                                           p.task_adversary_policy, obs))

    def get_deterministic_action(self, obs):
        return np.asarray(self._get_deterministic_action(self.state.params.policy, obs))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(jax.device_get(self.state), f)

    def load(self, path):
        with Path(path).open('rb') as f:
            self.state = jax.device_put(pickle.load(f))
