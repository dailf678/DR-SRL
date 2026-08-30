import argparse
import time
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import gym
import gymnasium as gym
import jax
from safe_env.register import register

from relax.algorithm.sac_rfsi_DR_s import SACRFSI_DR_S

from relax.network.sac_rfsi_DR_s import create_sac_rfsi_net_dr_s

from relax.trainer.off_policy import OffPolicyTrainer
from relax.utils.experience import Experience, Experience2
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.timing import catchtime

if __name__ == "__main__":

    alg = [
        "sac-rfsi-dr-s",
    ]
    ids = len(alg)
    for id in range(ids):
        num = 4
        for i in range(num):
            print(f"Training round {i + 1} / {num}...")
            parser = argparse.ArgumentParser()
            parser.add_argument("--alg", type=str, default=alg[id])
            parser.add_argument("--env", type=str, default="CartPole-v0")
            parser.add_argument("--hidden_num", type=int, default=2)
            parser.add_argument("--hidden_dim", type=int, default=128)
            parser.add_argument("--start_step", type=int, default=int(1e4))
            parser.add_argument("--total_step", type=int, default=int(6e5))
            parser.add_argument("--lr", type=float, default=3e-4)
            parser.add_argument("--certificate_lr", type=float, default=3e-4)
            parser.add_argument("--multiplier_lr", type=float, default=1e-4)
            parser.add_argument("--feasible_threshold",
                                type=float,
                                default=0.1)
            parser.add_argument("--infeasible_threshold",
                                type=float,
                                default=0.9)
            parser.add_argument("--seed", type=int, default=1)
            args = parser.parse_args()

            register()

            # Manage seeds
            master_seed = args.seed
            master_rng, _ = seeding(master_seed)
            env_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
                int, master_rng.integers(0, 2**32 - 1, 5))
            init_network_key = jax.random.PRNGKey(init_network_seed)
            train_key = jax.random.PRNGKey(train_seed)
            del init_network_seed, train_seed

            env = gym.make(args.env)
            eval_env = gym.make(args.env)

            obs_dim, act_dim = env.observation_space.shape[
                0], env.action_space.shape[0]
            hidden_sizes = [args.hidden_dim] * args.hidden_num
            barrier_input_dim = env.unwrapped.barrier_input_dim
            preprocess = env.unwrapped.preprocess

            buffer = TreeBuffer.from_experience(obs_dim,
                                                act_dim,
                                                size=args.total_step,
                                                seed=buffer_seed)

            if args.alg == "sac-rfsi-dr-s":
                agent, params = create_sac_rfsi_net_dr_s(
                    init_network_key,
                    obs_dim,
                    act_dim,
                    hidden_sizes,
                    barrier_input_dim=barrier_input_dim,
                    preprocess=preprocess,
                )
                algorithm = SACRFSI_DR_S(
                    agent,
                    params,
                    lr=args.lr,
                    certificate_lr=args.certificate_lr,
                    feasible_threshold=args.feasible_threshold,
                    infeasible_threshold=args.infeasible_threshold,
                    multiplier_lr=args.multiplier_lr,
                )
            else:
                raise ValueError(f"Invalid algorithm {args.alg}!")

            trainer = OffPolicyTrainer(
                env=env,
                algorithm=algorithm,
                buffer=buffer,
                start_step=args.start_step,
                total_step=args.total_step,
                evaluate_env=eval_env,
                log_path=PROJECT_ROOT / "logs" / args.env /
                (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") +
                 f'_s{args.seed}'),
            )

            # Warmup jit for more consistent timing
            @catchtime("warmup")
            def warmup_jit():
                dummy_key = jax.random.PRNGKey(0)
                dummy_data = jax.device_put(
                    Experience.create_example(obs_dim, act_dim,
                                              trainer.batch_size))
                dummy_state = jax.tree_util.tree_map(jax.numpy.copy,
                                                     algorithm.state)
                dummy_obs = env.observation_space.sample()
                dummy_state, _ = algorithm._update(dummy_key, dummy_state,
                                                   dummy_data)

                if args.alg == "sac-rfsi-dr-s":
                    algorithm._get_action(
                        dummy_key, dummy_state.params.policy,
                        dummy_state.params.safe_adversary_policy,
                        dummy_state.params.task_adversary_policy, dummy_obs)
                else:
                    print("warmup other algs")
                    algorithm._get_action(dummy_key, dummy_state.params.policy,
                                          dummy_obs)
                algorithm._get_deterministic_action(dummy_state.params.policy,
                                                    dummy_obs)

            warmup_jit()

            trainer.train(train_key)
