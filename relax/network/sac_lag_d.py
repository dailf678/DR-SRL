import math
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from relax.network.blocks import Activation, QNet, PolicyNet, DeterministicPolicyNet
from relax.network.common import WithSquashedGaussianPolicy
from numpyro.distributions import Normal


class SACLagParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    cost_q: hk.Params
    target_cost_q: hk.Params
    policy: hk.Params
    safe_adversary_policy: hk.Params
    task_adversary_policy: hk.Params
    log_alpha: jnp.ndarray
    multiplier_param: jnp.ndarray


@dataclass
class SACLagNet(WithSquashedGaussianPolicy):
    q: Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    target_entropy: float
    safe_adversary_policy: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]
    task_adversary_policy: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]

    def safe_adversary_policy_evaluate(self,
                                       safe_adversary_policy_params: hk.Params,
                                       obs: jnp.ndarray) -> jnp.ndarray:
        """for algorithm update"""
        z = self.safe_adversary_policy(safe_adversary_policy_params, obs)
        act = 0.2 * jnp.tanh(z)
        return act

    def task_adversary_policy_evaluate(self,
                                       task_adversary_policy_params: hk.Params,
                                       obs: jnp.ndarray) -> jnp.ndarray:
        """for algorithm update"""
        z = self.task_adversary_policy(task_adversary_policy_params, obs)
        act = 0.2 * jnp.tanh(z)
        return act

    def get_action_adv(self, key: jax.random.KeyArray,
                       policy_params: hk.Params,
                       safe_adv_policy_params: hk.Params,
                       task_adv_policy_params: hk.Params,
                       obs: jnp.ndarray) -> jnp.ndarray:
        """for data collection"""
        mean, std = self.policy(policy_params, obs)
        # obs_preprocess = self.preprocess(obs)
        mean_safe_adv = self.safe_adversary_policy(safe_adv_policy_params, obs)
        mean_task_adv = self.task_adversary_policy(task_adv_policy_params, obs)
        z = Normal(mean, std).sample(key)
        z_safe_adv = mean_safe_adv
        z_task_adv = mean_task_adv
        act = jnp.tanh(z)
        act_safe_adv = 0.2 * jnp.tanh(z_safe_adv)
        act_task_adv = 0.2 * jnp.tanh(z_task_adv)
        adv_choice = jax.random.bernoulli(key, p=0.5)
        act_adv = jnp.where(adv_choice, act_safe_adv, act_task_adv)
        return act + act_adv


def create_sac_lag_net_d(
    key: jax.random.KeyArray,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
) -> Tuple[SACLagNet, SACLagParams]:
    q = hk.without_apply_rng(
        hk.transform(lambda obs, act: QNet(hidden_sizes, activation)
                     (obs, act)))
    policy = hk.without_apply_rng(
        hk.transform(lambda obs: PolicyNet(act_dim, hidden_sizes, activation)
                     (obs)))
    safe_adversary_policy = hk.without_apply_rng(
        hk.transform(lambda obs: DeterministicPolicyNet(
            act_dim, hidden_sizes, activation)(obs)))
    task_adversary_policy = hk.without_apply_rng(
        hk.transform(lambda obs: DeterministicPolicyNet(
            act_dim, hidden_sizes, activation)(obs)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, cost_q_key, policy_key, safe_adversary_policy_key, task_adversary_policy_key = jax.random.split(
            key, 6)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        cost_q_params = q.init(cost_q_key, obs, act)
        target_cost_q_params = cost_q_params
        policy_params = policy.init(policy_key, obs)
        safe_adversary_policy_params = safe_adversary_policy.init(
            safe_adversary_policy_key, obs)
        task_adversary_policy_params = task_adversary_policy.init(
            task_adversary_policy_key, obs)
        log_alpha = jnp.array(0.0, dtype=jnp.float32)
        multiplier_param = jnp.array(math.log(math.exp(1) - 1),
                                     dtype=jnp.float32)
        return SACLagParams(
            q1=q1_params,
            q2=q2_params,
            target_q1=target_q1_params,
            target_q2=target_q2_params,
            cost_q=cost_q_params,
            target_cost_q=target_cost_q_params,
            policy=policy_params,
            safe_adversary_policy=safe_adversary_policy_params,
            task_adversary_policy=task_adversary_policy_params,
            log_alpha=log_alpha,
            multiplier_param=multiplier_param,
        )

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = SACLagNet(
        policy=policy.apply,
        q=q.apply,
        target_entropy=-act_dim,
        safe_adversary_policy=safe_adversary_policy.apply,
        task_adversary_policy=task_adversary_policy.apply,
    )
    return net, params
