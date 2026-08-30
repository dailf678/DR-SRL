import numpy as np
from safe_control_gym.envs.gym_control.cartpole import CartPole

from safe_env.base import BarrierEnv


class MyCartPole(CartPole, BarrierEnv):

    def step(self, action):
        feasible = self.check_goal_reached()
        infeasible = self.constraint_violated()
        obs, reward, done, info = super(MyCartPole, self).step(action)
        info.update({
            'cost': info['constraint_violation'],
            'feasible': feasible,
            'infeasible': infeasible,
        })
        return obs, reward, done, done, info

    def check_goal_reached(self):
        return bool(
            np.linalg.norm(self.state - self.X_GOAL) <
            self.TASK_INFO['stabilization_goal_tolerance'])

    def constraint_violated(self):
        c_value = self.constraints.get_values(self)
        return self.constraints.is_violated(self, c_value=c_value)
