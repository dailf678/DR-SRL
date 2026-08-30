from typing import NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from relax.algorithm.base import Algorithm
from relax.network.sac_rfsi_DR_s import SACFSINet, SACFSIParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class SACFSIOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState
    model: optax.OptState
    classifier: optax.OptState
    safe_policy: optax.OptState
    safe_adversary_policy: optax.OptState
    task_adversary_policy: optax.OptState
    barrier: optax.OptState
    multiplier: optax.OptState


class SACFSITrainState(NamedTuple):
    params: SACFSIParams
    opt_state: SACFSIOptStates
    step: int


class SACRFSI_DR_S(Algorithm):

    def __init__(
        self,
        agent: SACFSINet,
        params: SACFSIParams,
        *,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        certificate_lr: float = 3e-4,
        feasible_threshold: float = 0.1,
        infeasible_threshold: float = 0.9,
        lam: float = 0.1,
        eps: float = 0.01,
        multiplier_lr: float = 3e-4,
        multiplier_delay: int = 10,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.feasible_threshold = feasible_threshold
        self.infeasible_threshold = infeasible_threshold
        self.lam = lam
        self.eps = eps
        self.multiplier_delay = multiplier_delay
        self.name = "SACRFSI-DR-s"

        self.optim = optax.adam(lr)
        self.certificate_optim = optax.adam(certificate_lr)
        self.multiplier_optim = optax.adam(multiplier_lr)

        self.state = SACFSITrainState(
            params=params,
            opt_state=SACFSIOptStates(
                q1=self.certificate_optim.init(params.q1),
                q2=self.certificate_optim.init(params.q2),
                policy=self.optim.init(params.policy),
                log_alpha=self.optim.init(params.log_alpha),
                model=self.certificate_optim.init(params.model),
                classifier=self.certificate_optim.init(params.classifier),
                safe_policy=self.optim.init(params.safe_policy),
                safe_adversary_policy=self.optim.init(
                    params.safe_adversary_policy),
                task_adversary_policy=self.optim.init(
                    params.task_adversary_policy),
                barrier=self.certificate_optim.init(params.barrier),
                multiplier=self.multiplier_optim.init(params.multiplier),
            ),
            step=0,
        )

        @jax.jit
        def stateless_update(
            key: jax.random.KeyArray,
            state: SACFSITrainState,
            data: Experience,
        ) -> Tuple[SACFSITrainState, Metric]:
            obs, action, next_obs, reward, done, feasible, infeasible = (
                data.obs,
                data.action,
                data.next_obs,
                data.reward,
                data.done,
                data.feasible,
                data.infeasible,
            )
            (
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
            ) = state.params
            (
                q1_opt_state,
                q2_opt_state,
                policy_opt_state,
                log_alpha_opt_state,
                model_opt_state,
                classifier_opt_state,
                safe_policy_opt_state,
                safe_adversary_policy_opt_state,
                task_adversary_policy_opt_state,
                barrier_opt_state,
                multiplier_opt_state,
            ) = state.opt_state
            step = state.step
            next_eval_key, new_eval_key = jax.random.split(key)

            obs_preprocess = self.agent.preprocess(obs)
            next_obs_preprocess = self.agent.preprocess(next_obs)

            # update model
            def model_loss_fn(model_params: hk.Params):
                next_obs_pred = obs_preprocess + self.agent.model(
                    model_params, obs_preprocess, action)
                model_loss = jnp.mean((next_obs_preprocess - next_obs_pred)**2)
                return model_loss

            model_loss, model_grads = jax.value_and_grad(model_loss_fn)(
                model_params)
            model_updates, model_opt_state = self.certificate_optim.update(
                model_grads, model_opt_state)
            model_params = optax.apply_updates(model_params, model_updates)

            # update classifier
            def classifier_loss_fn(classifier_params: hk.Params):
                logits = self.agent.classifier(classifier_params,
                                               obs_preprocess)
                labeled_target = 0 * feasible + 1 * infeasible
                labeled = feasible | infeasible
                supervised_loss = jnp.mean(
                    optax.sigmoid_binary_cross_entropy(logits, labeled_target)
                    * labeled)

                new_action_u_safe = self.agent.safe_policy_evaluate(
                    safe_policy_params, obs_preprocess)
                new_action_a_safe = self.agent.safe_adversary_policy_evaluate(
                    safe_adversary_policy_params, obs_preprocess)
                new_action = new_action_u_safe + new_action_a_safe
                new_next_obs_obs_preprocess = obs_preprocess + self.agent.model(
                    model_params, obs_preprocess, new_action)
                next_probs = jax.nn.sigmoid(
                    self.agent.classifier(target_classifier_params,
                                          new_next_obs_obs_preprocess))
                unlabeled_target = self.gamma * jax.lax.stop_gradient(
                    next_probs)
                unsupervised_loss = jnp.mean(
                    optax.sigmoid_binary_cross_entropy(
                        logits, unlabeled_target) * (1 - labeled))
                classifier_loss = supervised_loss + unsupervised_loss
                return classifier_loss, (
                    supervised_loss,
                    unsupervised_loss,
                    logits,
                    new_next_obs_obs_preprocess,
                )

            (classifier_loss, aux), classifier_grads = \
                jax.value_and_grad(classifier_loss_fn, has_aux=True)(classifier_params)
            supervised_loss, unsupervised_loss, logits, new_next_obs = aux
            classifier_updates, classifier_opt_state = \
                self.certificate_optim.update(classifier_grads, classifier_opt_state)
            classifier_params = optax.apply_updates(classifier_params,
                                                    classifier_updates)

            # update barrier
            def barrier_loss_fn(barrier_params: hk.Params):
                barrier = self.agent.barrier(barrier_params, obs_preprocess)
                next_barrier = self.agent.barrier(barrier_params, new_next_obs)

                def get_grad_norm(params, x):
                    grad_fn = jax.grad(
                        lambda _x: jnp.sum(self.agent.barrier(params, _x)))
                    grads = grad_fn(x)
                    return jnp.linalg.norm(grads, axis=1)

                ########################### Prevent excessive gradient
                grad_norm_b = get_grad_norm(barrier_params, obs_preprocess)
                target_lip_const = 10
                eta = -1.0
                lip_loss_val = jax.nn.relu(
                    jax.nn.relu(-(barrier + eta)).squeeze() *
                    (grad_norm_b - target_lip_const))
                lipschitz_loss = jnp.mean(lip_loss_val)

                probs = jax.nn.sigmoid(logits)
                classifier_feasible = feasible | (
                    ~infeasible & (probs < self.feasible_threshold))
                classifier_infeasible = infeasible | (
                    ~feasible & (probs > self.infeasible_threshold))

                feasible_loss = jnp.mean(
                    jnp.maximum(self.eps + barrier, 0) * classifier_feasible)
                infeasible_loss = jnp.mean(
                    jnp.maximum(self.eps - barrier, 0) * classifier_infeasible)
                invariant_loss = jnp.mean(
                    jnp.maximum(
                        self.eps + next_barrier - (1 - self.lam) * barrier, 0))

                barrier_loss = feasible_loss + infeasible_loss + invariant_loss + 0.04 * lipschitz_loss
                return barrier_loss, (
                    feasible_loss,
                    infeasible_loss,
                    invariant_loss,
                    classifier_feasible,
                    classifier_infeasible,
                )

            (barrier_loss, aux), barrier_grads = jax.value_and_grad(
                barrier_loss_fn, has_aux=True)(barrier_params)
            feasible_loss, infeasible_loss, invariant_loss, classifier_feasible, classifier_infeasible = aux
            barrier_updates, barrier_opt_state = self.certificate_optim.update(
                barrier_grads, barrier_opt_state)
            barrier_params = optax.apply_updates(barrier_params,
                                                 barrier_updates)

            def safe_adversary_policy_loss_fn(
                    safe_adversary_policy_params: hk.Params):
                new_action_a = self.agent.safe_adversary_policy_evaluate(
                    safe_adversary_policy_params, obs_preprocess)
                new_action_u_safe = self.agent.safe_policy_evaluate(
                    safe_policy_params, obs_preprocess)
                new_action = new_action_u_safe + new_action_a
                new_next_obs_obs_preprocess = obs_preprocess + self.agent.model(
                    model_params, obs_preprocess, new_action)
                next_probs = jax.nn.sigmoid(
                    self.agent.classifier(classifier_params,
                                          new_next_obs_obs_preprocess))
                safe_adversary_policy_loss = -jnp.mean(next_probs)
                return safe_adversary_policy_loss

            safe_adversary_policy_loss, safe_adversary_policy_grads = jax.value_and_grad(
                safe_adversary_policy_loss_fn)(safe_adversary_policy_params)
            safe_adversary_policy_updates, safe_adversary_policy_opt_state = \
                self.optim.update(safe_adversary_policy_grads, safe_adversary_policy_opt_state)
            safe_adversary_policy_params = optax.apply_updates(
                safe_adversary_policy_params, safe_adversary_policy_updates)

            # update safe policy
            def safe_policy_loss_fn(safe_policy_params: hk.Params):
                new_action_u_safe = self.agent.safe_policy_evaluate(
                    safe_policy_params, obs_preprocess)
                new_action_a = self.agent.safe_adversary_policy_evaluate(
                    safe_adversary_policy_params, obs_preprocess)
                new_action = new_action_u_safe + new_action_a
                new_next_obs_obs_preprocess = obs_preprocess + self.agent.model(
                    model_params, obs_preprocess, new_action)
                next_probs = jax.nn.sigmoid(
                    self.agent.classifier(classifier_params,
                                          new_next_obs_obs_preprocess))
                safe_policy_loss = jnp.mean(next_probs)
                return safe_policy_loss

            safe_policy_loss, safe_policy_grads = jax.value_and_grad(
                safe_policy_loss_fn)(safe_policy_params)
            safe_policy_updates, safe_policy_opt_state = \
                self.optim.update(safe_policy_grads, safe_policy_opt_state)
            safe_policy_params = optax.apply_updates(safe_policy_params,
                                                     safe_policy_updates)

            # update q
            next_action_u_task, next_logp = self.agent.evaluate(
                next_eval_key, policy_params, next_obs)
            next_action_a_task = self.agent.task_adversary_policy_evaluate(
                task_adversary_policy_params, next_obs)
            next_action = next_action_u_task + next_action_a_task
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target,
                                   q2_target) - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params):
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup)**2)
                return q_loss, q

            (q1_loss,
             q1), q1_grads = jax.value_and_grad(q_loss_fn,
                                                has_aux=True)(q1_params)
            (q2_loss,
             q2), q2_grads = jax.value_and_grad(q_loss_fn,
                                                has_aux=True)(q2_params)
            q1_update, q1_opt_state = self.certificate_optim.update(
                q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.certificate_optim.update(
                q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            def task_adversary_policy_loss_fn(
                    task_adversary_policy_params: hk.Params):
                new_action_a_task = self.agent.task_adversary_policy_evaluate(
                    task_adversary_policy_params, obs)
                new_action_u_task, new_logp = self.agent.evaluate(
                    new_eval_key, policy_params, obs)
                new_action = new_action_u_task + new_action_a_task
                q1 = self.agent.q(q1_params, obs, new_action)
                q2 = self.agent.q(q2_params, obs, new_action)
                q = jnp.minimum(q1, q2)
                task_adversary_policy_loss = -jnp.mean(
                    jnp.exp(log_alpha) * new_logp - q)
                return task_adversary_policy_loss

            task_adversary_policy_loss, task_adversary_policy_grads = jax.value_and_grad(
                task_adversary_policy_loss_fn)(task_adversary_policy_params)
            task_adversary_policy_updates, task_adversary_policy_opt_state = \
                self.optim.update(
                    task_adversary_policy_grads, task_adversary_policy_opt_state)
            task_adversary_policy_params = optax.apply_updates(
                task_adversary_policy_params, task_adversary_policy_updates)

            def estimate_local_lipschitz(key, obs, action):
                K = 10
                a_max = 0.2
                batch_size = obs.shape[0]
                action_dim = action.shape[-1]
                state_dim = obs.shape[-1]
                key_dir, key_r = jax.random.split(key)
                directions = jax.random.normal(key_dir, (K, action_dim))
                directions = directions / (
                    jnp.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
                radius = a_max * (jax.random.uniform(key_r, (K, 1))
                                  **(1.0 / action_dim))
                disturbances = radius * directions
                sampled_actions = (action[:, None, :] +
                                   disturbances[None, :, :])

                sampled_obs = jnp.broadcast_to(obs[:, None, :],
                                               (batch_size, K, state_dim))
                flat_obs = sampled_obs.reshape(-1, state_dim)
                flat_action = sampled_actions.reshape(-1, action_dim)
                next_obs = flat_obs + self.agent.model(model_params, flat_obs,
                                                       flat_action)
                grad_fn = jax.grad(
                    lambda x: jnp.sum(self.agent.barrier(barrier_params, x)))
                grad_b = grad_fn(next_obs)
                grad_norm = jnp.linalg.norm(grad_b,
                                            axis=1).reshape(batch_size, K)
                return jnp.max(grad_norm, axis=1)

            # update policy
            def policy_loss_fn(policy_params: hk.Params):
                policy_key, lip_key = jax.random.split(new_eval_key)
                new_action_u_task, new_logp = self.agent.evaluate(
                    policy_key, policy_params, obs)
                new_action_a_task = self.agent.task_adversary_policy_evaluate(
                    task_adversary_policy_params, obs)
                new_action = new_action_u_task + new_action_a_task
                q1 = self.agent.q(q1_params, obs, new_action)
                q2 = self.agent.q(q2_params, obs, new_action)
                q = jnp.minimum(q1, q2)

                #################e lip
                local_lip = estimate_local_lipschitz(lip_key, obs_preprocess,
                                                     new_action_u_task)
                robust_term = 0.02 * 1.22 * local_lip * 0.2  #0.02 simulator sample time，1.22 ||d|| adversarial input matrix, 0.2 disturbance.

                new_action = new_action_u_task
                barrier = self.agent.barrier(barrier_params, obs_preprocess)
                new_next_obs_obs_preprocess = obs_preprocess + self.agent.model(
                    model_params, obs_preprocess, new_action)
                next_barrier = self.agent.barrier(barrier_params,
                                                  new_next_obs_obs_preprocess)
                stable_barrier = jnp.maximum(
                    jnp.maximum(jnp.abs(barrier), jnp.abs(next_barrier)), 1e-2)
                raw_violation = (self.eps + next_barrier -
                                 (1 - self.lam) * barrier +
                                 robust_term) / stable_barrier
                multiplier_val = self.agent.multiplier(multiplier_params,
                                                       obs_preprocess)
                policy_loss = jnp.mean(
                    jnp.exp(log_alpha) * new_logp - q +
                    multiplier_val * raw_violation)
                return policy_loss, (new_logp, robust_term, multiplier_val,
                                     raw_violation)

            (policy_loss, aux), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True)(policy_params)
            new_logp, robust_term, multiplier_val_1, raw_violation = aux
            policy_update, policy_opt_state = self.optim.update(
                policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jnp.ndarray) -> jnp.ndarray:
                log_alpha_loss = -jnp.mean(
                    log_alpha * (new_logp + self.agent.target_entropy))
                return log_alpha_loss

            log_alpha_grads = jax.grad(log_alpha_loss_fn)(log_alpha)
            log_alpha_update, log_alpha_opt_state = self.optim.update(
                log_alpha_grads, log_alpha_opt_state)
            log_alpha = optax.apply_updates(log_alpha, log_alpha_update)

            # update multiplier
            def multiplier_loss_fn(
                    multiplier_params: hk.Params) -> jnp.ndarray:
                multiplier_val = self.agent.multiplier(multiplier_params,
                                                       obs_preprocess)
                multiplier_loss = -jnp.mean(
                    multiplier_val * jax.lax.stop_gradient(raw_violation))
                return multiplier_loss

            def multiplier_update_fn(
                multiplier_params: hk.Params,
                multiplier_opt_state: optax.OptState
            ) -> Tuple[hk.Params, optax.OptState]:
                multiplier_grads = jax.grad(multiplier_loss_fn)(
                    multiplier_params)
                multiplier_update, multiplier_opt_state = self.multiplier_optim.update(
                    multiplier_grads, multiplier_opt_state)
                multiplier_params = optax.apply_updates(
                    multiplier_params, multiplier_update)
                return multiplier_params, multiplier_opt_state

            multiplier_params, multiplier_opt_state = jax.lax.cond(
                step % self.multiplier_delay == 0,
                multiplier_update_fn,
                lambda p, s: (p, s),
                multiplier_params,
                multiplier_opt_state,
            )

            # update target networks
            target_classifier_params = optax.incremental_update(
                classifier_params, target_classifier_params, self.tau)
            target_q1_params = optax.incremental_update(
                q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(
                q2_params, target_q2_params, self.tau)

            state = SACFSITrainState(
                params=SACFSIParams(
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
                ),
                opt_state=SACFSIOptStates(
                    q1_opt_state,
                    q2_opt_state,
                    policy_opt_state,
                    log_alpha_opt_state,
                    model_opt_state,
                    classifier_opt_state,
                    safe_policy_opt_state,
                    safe_adversary_policy_opt_state,
                    task_adversary_policy_opt_state,
                    barrier_opt_state,
                    multiplier_opt_state,
                ),
                step=step + 1,
            )
            info = {
                "model_loss":
                model_loss,
                "supervised_loss":
                supervised_loss,
                "unsupervised_loss":
                unsupervised_loss,
                "classifier_loss":
                classifier_loss,
                "feasible_loss":
                feasible_loss,
                "infeasible_loss":
                infeasible_loss,
                "invariant_loss":
                invariant_loss,
                "barrier_loss":
                barrier_loss,
                "label_feasible_ratio":
                jnp.mean(feasible),
                "label_infeasible_ratio":
                jnp.mean(infeasible),
                "classifier_feasible_ratio":
                jnp.mean(classifier_feasible),
                "classifier_infeasible_ratio":
                jnp.mean(classifier_infeasible),
                "safe_policy_loss":
                safe_policy_loss,
                "safe_adversary_policy_loss":
                safe_adversary_policy_loss,
                "task_adversary_policy_loss":
                task_adversary_policy_loss,
                "q1_loss":
                q1_loss,
                "q2_loss":
                q2_loss,
                "q1":
                jnp.mean(q1),
                "q2":
                jnp.mean(q2),
                "policy_loss":
                policy_loss,
                "entropy":
                -jnp.mean(new_logp),
                "alpha":
                jnp.exp(log_alpha),
                "robust_term":
                jnp.mean(robust_term),
                "max robust_term":
                jnp.max(robust_term),
                "multiplier_val":
                jnp.mean(multiplier_val_1),
                "max_multiplier_val":
                jnp.max(multiplier_val_1),
                "raw_violation":
                jnp.mean(raw_violation),
                "raw_violation_max":
                jnp.max(raw_violation),
                "raw_violation_min":
                jnp.min(raw_violation),
                "positive_violation_ratio":
                jnp.mean((raw_violation > 0).astype(jnp.float32)),
                "positive_violation_mean":
                jnp.sum(jnp.maximum(raw_violation, 0.0)) /
                (jnp.sum(raw_violation > 0) + 1e-6),
            }
            return state, info

        self._implement_common_behavior(stateless_update,
                                        self.agent.get_action_adv,
                                        self.agent.get_deterministic_action)
