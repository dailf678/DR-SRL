import math


from dataclasses import dataclass


from typing import Callable, Tuple


import jax, jax.numpy as jnp


import haiku as hk


from numpyro.distributions import Normal


@dataclass
class WithSquashedGaussianPolicy:
    policy: Callable[[hk.Params, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]

    def get_action(self, key: jax.random.KeyArray, policy_params: hk.Params,
                   obs: jnp.ndarray) -> jnp.ndarray:
        """for data collection"""
        mean, std = self.policy(policy_params, obs)
        z = Normal(mean, std).sample(key)
        act = jnp.tanh(z)
        return act

    def get_deterministic_action(self, policy_params: hk.Params,
                                 obs: jnp.ndarray) -> jnp.ndarray:
        """for evaluation"""
        mean, _ = self.policy(policy_params, obs)
        act = jnp.tanh(mean)
        return act

    def evaluate(self, key: jax.random.KeyArray, policy_params: hk.Params,
                 obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """for algorithm update"""
        mean, std = self.policy(policy_params, obs)
        dist = Normal(mean, std)
        z = dist.rsample(key)
        act = jnp.tanh(z)
        logp = (dist.log_prob(z) - 2 *
                (math.log(2) - z - jax.nn.softplus(-2 * z))).sum(axis=-1)
        return act, logp


@dataclass
class WithSquashedGaussianPolicy_ts:
    policy: Callable[[hk.Params, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
    safe_policy: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]
    safe_adversary_policy: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]
    task_adversary_policy: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]

    def get_action(self, key: jax.random.KeyArray, policy_params: hk.Params,
                   obs: jnp.ndarray) -> jnp.ndarray:
        """for data collection"""
        mean, std = self.policy(policy_params, obs)
        z = Normal(mean, std).sample(key)
        act = jnp.tanh(z)
        return act


    def get_deterministic_action(self, policy_params: hk.Params,
                                 obs: jnp.ndarray) -> jnp.ndarray:
        """for evaluation"""
        mean, _ = self.policy(policy_params, obs)
        act = jnp.tanh(mean)
        return act


    def evaluate(self, key: jax.random.KeyArray, policy_params: hk.Params,
                 obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """for algorithm update"""
        mean, std = self.policy(policy_params, obs)
        dist = Normal(mean, std)
        z = dist.rsample(key)
        act = jnp.tanh(z)
        logp = (dist.log_prob(z) - 2 *
                (math.log(2) - z - jax.nn.softplus(-2 * z))).sum(axis=-1)
        return act, logp

