"""Network construction shared by training and checkpoint validation."""
import jax
from relax.algorithm.sac_rfsi_DR_s import SACRFSI_DR_S
from relax.algorithm.sac_fsi_DR import SACFSI_DR
from relax.algorithm.sac_fsi_A import SACFSI_A
from relax.algorithm.sac_lag_d import SACLag_D
from relax.network.sac_rfsi_DR_s import create_sac_rfsi_net_dr_s
from relax.network.sac_fsi_DR import create_sac_fsi_net_dr
from relax.network.sac_fsi_A import create_sac_fsi_net_a
from relax.network.sac_lag_d import create_sac_lag_net_d

METHODS = {
    'sac-rfsi-dr-s': (create_sac_rfsi_net_dr_s, SACRFSI_DR_S),
    'sac-fsi-dr': (create_sac_fsi_net_dr, SACFSI_DR),
    'sac-fsi-a': (create_sac_fsi_net_a, SACFSI_A),
    'sac-lag-d': (create_sac_lag_net_d, SACLag_D),
}


def build_algorithm(args, env, seed):
    net_fn, alg_class = METHODS[args.alg]
    net_kwargs = {}
    if args.alg != 'sac-lag-d':
        net_kwargs.update(barrier_input_dim=env.unwrapped.barrier_input_dim,
                          preprocess=env.unwrapped.preprocess)
    agent, params = net_fn(jax.random.PRNGKey(seed), env.observation_space.shape[0],
                           env.action_space.shape[0], [args.hidden_dim] * args.hidden_num,
                           **net_kwargs)
    kwargs = dict(lr=args.lr, multiplier_lr=args.multiplier_lr)
    if args.alg != 'sac-lag-d':
        kwargs.update(certificate_lr=args.certificate_lr,
                      feasible_threshold=args.feasible_threshold,
                      infeasible_threshold=args.infeasible_threshold)
    return alg_class(agent, params, **kwargs)
