"""Safety labels and Gymnasium API for the original PyBullet CartPole."""
import numpy as np
from safe_control_gym.envs.gym_control.cartpole import CartPole
from safe_env.base import BarrierEnv


class MyCartPole(CartPole, BarrierEnv):
    def reset(self, *, seed=None, options=None):
        # Upstream goal termination can return before setting this info flag.
        self.out_of_bounds = False
        obs, info = super().reset(seed=seed)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        # Published feasible/infeasible labels refer to the pre-step state.
        feasible = self.check_goal_reached()
        infeasible = self.constraint_violated()
        obs, reward, done, info = super().step(action)
        truncated = bool(info.get('TimeLimit.truncated', False))
        info.update(cost=info['constraint_violation'], feasible=feasible,
                    infeasible=infeasible)
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done and not truncated), truncated, info

    def check_goal_reached(self):
        return bool(np.linalg.norm(self.state - self.X_GOAL) <
                    self.TASK_INFO['stabilization_goal_tolerance'])

    def constraint_violated(self):
        return self.constraints.is_violated(self, c_value=self.constraints.get_values(self))
