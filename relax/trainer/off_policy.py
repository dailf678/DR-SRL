"""Single-environment off-policy sampling, logging and checkpointing."""
import json
from pathlib import Path
import jax
import numpy as np
from tensorboardX import SummaryWriter
from relax.utils.experience import Experience


class OffPolicyTrainer:
    def __init__(self, env, algorithm, buffer, log_path, batch_size=256,
                 start_step=10000, total_step=600000, save_every=10000,
                 log_every=1000):
        self.env, self.algorithm, self.buffer = env, algorithm, buffer
        self.log_path = Path(log_path)
        self.batch_size, self.start_step, self.total_step = batch_size, start_step, total_step
        self.save_every, self.log_every = save_every, log_every

    def train(self, key):
        writer = SummaryWriter(str(self.log_path))
        obs, _ = self.env.reset()
        ep_ret = ep_cost = 0.0
        ep_len = episodes = updates = 0
        last_metrics = {}
        accumulated = {}
        try:
            for step in range(1, self.total_step + 1):
                sample_key, update_key = jax.random.split(jax.random.fold_in(key, step - 1))
                action = (self.env.action_space.sample() if step <= self.start_step
                          else self.algorithm.get_action(sample_key, obs))
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                cost = float(info.get('cost', 0.0))
                # Preserve the original replay done flag, including time limit.
                self.buffer.add(Experience(obs, action, reward, cost, next_obs, done,
                                           info.get('feasible', False), info.get('infeasible', False),
                                           info.get('barrier', 0.0), info.get('next_barrier', 0.0)))
                ep_ret += reward
                ep_cost += cost
                ep_len += 1
                if done:
                    episodes += 1
                    for name, value in [('episode_return', ep_ret), ('episode_cost', ep_cost),
                                        ('episode_length', ep_len)]:
                        writer.add_scalar('sample/' + name, value, step)
                    obs, _ = self.env.reset()
                    ep_ret = ep_cost = 0.0
                    ep_len = 0
                else:
                    obs = next_obs
                if step >= self.start_step:
                    last_metrics = self.algorithm.update(update_key, self.buffer.sample(self.batch_size))
                    if not all(np.isfinite(v) for v in last_metrics.values()):
                        raise FloatingPointError(f'Non-finite training metrics at step {step}')
                    updates += 1
                    for name, value in last_metrics.items():
                        accumulated.setdefault(name, []).append(value)
                if step % self.log_every == 0:
                    for name, values in accumulated.items():
                        writer.add_scalar('update/' + name, np.mean(values), step)
                    accumulated.clear()
                    print(f'step={step} updates={updates} episodes={episodes}', flush=True)
                if step % self.save_every == 0:
                    self.algorithm.save(self.log_path / f'state_{step}.pkl')
            self.algorithm.save(self.log_path / 'state.pkl')
            summary = dict(steps=self.total_step, updates=updates, episodes=episodes,
                           final_metrics=last_metrics)
            (self.log_path / 'summary.json').write_text(json.dumps(summary, indent=2))
            return summary
        finally:
            writer.close()
