import math
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from relax.network.blocks import Activation, QNet, PolicyNet, ValueNet, ModelNet, DeterministicPolicyNet
from relax.network.common import WithSquashedGaussianPolicy, WithSquashedGaussianPolicy_ts
from numpyro.distributions import Normal


class SACFSIParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    log_alpha: jnp.ndarray
    model: hk.Params
    classifier: hk.Params
    target_classifier: hk.Params
    safe_policy: hk.Params
    safe_adversary_policy: hk.Params
    task_adversary_policy: hk.Params
    barrier: hk.Params
    multiplier: hk.Params


@dataclass
class SACFSINet(WithSquashedGaussianPolicy_ts):
    q: Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    model: Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    classifier: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]
    safe_policy: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]
    safe_adversary_policy: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]
    task_adversary_policy: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]
    barrier: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]
    target_entropy: float
    preprocess: Callable[[jnp.ndarray], jnp.ndarray]
    multiplier: Callable[[hk.Params, jnp.ndarray], jnp.ndarray]

    def safe_policy_evaluate(self, safe_policy_params: hk.Params,
                             obs: jnp.ndarray) -> jnp.ndarray:
        """for algorithm update"""
        z = self.safe_policy(safe_policy_params, obs)
        act = jnp.tanh(z)
        return act

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
        obs_preprocess = self.preprocess(obs)
        mean_safe_adv = self.safe_adversary_policy(safe_adv_policy_params,
                                                   obs_preprocess)
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


def create_sac_rfsi_net_dr_s(
    key: jax.random.KeyArray,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    barrier_input_dim: int,
    preprocess: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    activation: Activation = jax.nn.relu,
) -> Tuple[SACFSINet, SACFSIParams]:
    q = hk.without_apply_rng(
        hk.transform(lambda obs, act: QNet(hidden_sizes, activation)
                     (obs, act)))
    policy = hk.without_apply_rng(
        hk.transform(lambda obs: PolicyNet(act_dim, hidden_sizes, activation)
                     (obs)))
    model = hk.without_apply_rng(
        hk.transform(lambda obs, act: ModelNet(barrier_input_dim, hidden_sizes,
                                               activation)(obs, act)))
    classifier = hk.without_apply_rng(
        hk.transform(lambda obs: ValueNet(hidden_sizes, jax.nn.elu)(obs)))
    safe_policy = hk.without_apply_rng(
        hk.transform(lambda obs: DeterministicPolicyNet(
            act_dim, hidden_sizes, activation)(obs)))
    safe_adversary_policy = hk.without_apply_rng(
        hk.transform(lambda obs: DeterministicPolicyNet(
            act_dim, hidden_sizes, activation)(obs)))
    task_adversary_policy = hk.without_apply_rng(
        hk.transform(lambda obs: DeterministicPolicyNet(
            act_dim, hidden_sizes, activation)(obs)))
    barrier = hk.without_apply_rng(
        hk.transform(lambda obs: ValueNet(hidden_sizes, jax.nn.elu)(obs)))

    def multiplier_net_fn(obs):
        x = hk.nets.MLP(list(hidden_sizes), activation=activation)(obs)
        init_val = 1.0  # 你可以根据环境难度调整初始乘子大小 (例如 0.1 或 1.0)
        init_bias = math.log(math.exp(init_val) - 1.0 + 1e-8)
        x = hk.Linear(1,
                      w_init=hk.initializers.RandomUniform(-3e-3, 3e-3),
                      b_init=hk.initializers.Constant(init_bias))(x)
        multiplier_out = jax.nn.softplus(x)
        multiplier_out = jnp.clip(multiplier_out, 0.0, 50.0)
        return jnp.squeeze(multiplier_out, axis=-1)

    multiplier = hk.without_apply_rng(hk.transform(multiplier_net_fn))

    @jax.jit
    def init(key, obs, act, barrier_obs):
        q1_key, q2_key, policy_key, model_key, classifier_key, safe_policy_key, safe_adversary_policy_key, task_adversary_policy_key, barrier_key, multiplier_key = jax.random.split(
            key, 10)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs)
        log_alpha = jnp.array(0.0, dtype=jnp.float32)
        model_params = model.init(model_key, barrier_obs, act)
        classifier_params = classifier.init(classifier_key, barrier_obs)
        target_classifier_params = classifier_params
        safe_policy_params = safe_policy.init(safe_policy_key, barrier_obs)
        safe_adversary_policy_params = safe_adversary_policy.init(
            safe_adversary_policy_key, barrier_obs)
        task_adversary_policy_params = task_adversary_policy.init(
            task_adversary_policy_key, obs)
        barrier_params = barrier.init(barrier_key, barrier_obs)
        multiplier_params = multiplier.init(multiplier_key, barrier_obs)
        return SACFSIParams(
            q1_params,
            q2_params,
            target_q1_params,
            target_q2_params,
            policy_params,
            log_alpha,
            model_params,
            classifier_params,
            target_classifier_params,
            safe_policy_params,
            safe_adversary_policy_params,
            task_adversary_policy_params,
            barrier_params,
            multiplier_params,
        )

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    sample_barrier_obs = jnp.zeros((1, barrier_input_dim))
    params = init(key, sample_obs, sample_act, sample_barrier_obs)

    net = SACFSINet(
        policy=policy.apply,
        q=q.apply,
        model=model.apply,
        classifier=classifier.apply,
        safe_policy=safe_policy.apply,
        safe_adversary_policy=safe_adversary_policy.apply,
        task_adversary_policy=task_adversary_policy.apply,
        barrier=barrier.apply,
        target_entropy=-act_dim,
        preprocess=preprocess,
        multiplier=multiplier.apply,
    )
    return net, params
