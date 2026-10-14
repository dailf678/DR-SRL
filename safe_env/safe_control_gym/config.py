cartpole_config = {
    'ctrl_freq':
    50,
    'pyb_freq':
    50,
    'episode_len_sec':
    5,
    'normalized_rl_action_space':
    True,

    # task
    'task':
    'stabilization',
    'task_info': {
        'stabilization_goal': [0],
        'stabilization_goal_tolerance': 0.005,
    },
    'cost':
    'rl_reward',

    # init
    'randomized_init':
    True,
    'init_state_randomization_info': {
        'init_x': {
            'distrib': 'uniform',
            'low': -0.5,
            'high': 0.5
        },
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.1,
            'high': 0.1
            # 'low': -0.19,
            # 'high': 0.19
        },
    },

    # constraint
    'constraints': [{
        'constraint_form': 'bounded_constraint',
        'constrained_variable': 'state',
        'active_dims': [2, 3],
        'lower_bounds': [-0.2, -0.2],
        'upper_bounds': [0.2, 0.2],
        # 'active_dims': [2],
        # 'lower_bounds': [-0.2],
        # 'upper_bounds': [0.2],
    }],
    'done_on_violation':
    False,

    # custom
    'rew_state_weight': [1, 0, 0, 0],
    'rew_act_weight':
    0.001,
    'rew_exponential':
    True,
    'done_on_out_of_bound':
    True,
}
