"""Training entry point.

    python main.py --env ALE/MsPacman-v5 --variant dqn --training-steps 10000000

Every hyper-parameter is a CLI flag; the resolved config is written to
`runs/<run_name>/config.json` so a run is always reproducible from its log dir.
"""

import argparse
import os
import sys

import torch

# One thread per process: 4-5 trainings run side by side, and letting each
# spawn 20 OpenMP threads makes them all slower than letting each have one.
torch.set_num_threads(1)

from dqn_agent_atari import AtariDQNAgent
from utils import make_run_name, save_config, seed_everything, tee_stdout

VARIANTS = ("dqn", "ddqn", "dueling", "dueling_ddqn")


def build_parser():
	p = argparse.ArgumentParser(description="DQN / DDQN / Dueling / parallel rollout on Atari")
	p.add_argument("--env", default="ALE/MsPacman-v5", help="gymnasium env id")
	p.add_argument("--variant", default="dqn", choices=VARIANTS)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--run-name", default=None, help="default: <env>_<variant>_s<seed>_<MMDD-HHMM>")
	p.add_argument("--run-root", default=None, help="parent of runs/ and checkpoints/ (default: repo root)")

	g = p.add_argument_group("schedule")
	g.add_argument("--training-steps", type=float, default=10_000_000)
	g.add_argument("--warmup-steps", type=int, default=20_000)
	g.add_argument("--eps-decay", type=int, default=1_000_000, help="env steps from 1.0 to eps-min")
	g.add_argument("--eps-min", type=float, default=0.1)
	g.add_argument("--update-freq", type=int, default=4, help="env steps per gradient step")
	g.add_argument("--update-target-freq", type=int, default=10_000)

	g = p.add_argument_group("optimisation")
	g.add_argument("--batch-size", type=int, default=32)
	g.add_argument("--learning-rate", type=float, default=0.0000625)
	g.add_argument("--gamma", type=float, default=0.99)
	g.add_argument("--loss", default="mse", choices=("mse", "huber"))
	g.add_argument("--max-grad-norm", type=float, default=10.0, help="0 disables clipping")
	g.add_argument("--replay-buffer-capacity", type=int, default=100_000)

	g = p.add_argument_group("environment")
	g.add_argument("--num-envs", type=int, default=1, help=">1 enables parallelized rollout")
	g.add_argument("--sync-envs", action="store_true", help="SyncVectorEnv instead of Async")
	g.add_argument("--no-clip-reward", dest="clip_reward", action="store_false",
	               help="train on raw rewards (the raw score is logged either way)")
	g.add_argument("--life-loss-terminal", action="store_true",
	               help="treat a lost life as terminal in the TD target only; the env is "
	                    "not reset and evaluation always plays full games")

	g = p.add_argument_group("evaluation / logging")
	g.add_argument("--eval-interval", type=int, default=100_000, help="env steps between evaluations")
	g.add_argument("--eval-episode", type=int, default=5)
	g.add_argument("--eval-epsilon", type=float, default=0.01)
	g.add_argument("--eval-seed-base", type=int, default=10_000)
	g.add_argument("--log-interval", type=int, default=100, help="updates per Loss/Q-Mean flush")

	p.add_argument("--resume", default=None, help="path to a checkpoint to continue from")
	p.add_argument("--cpu", action="store_true")
	return p


def main(argv=None):
	args = build_parser().parse_args(argv)

	repo_root = args.run_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	run_name = args.run_name or make_run_name(args.env, args.variant, args.seed)
	logdir = os.path.join(repo_root, "runs", run_name)
	ckptdir = os.path.join(repo_root, "checkpoints", run_name)
	os.makedirs(logdir, exist_ok=True)
	os.makedirs(ckptdir, exist_ok=True)
	tee_stdout(os.path.join(logdir, "train.log"))

	seed_everything(args.seed)

	config = {
		"gpu": not args.cpu,
		"training_steps": int(args.training_steps),
		"gamma": args.gamma,
		"batch_size": args.batch_size,
		"eps_min": args.eps_min,
		"warmup_steps": args.warmup_steps,
		"eps_decay": args.eps_decay,
		"eval_epsilon": args.eval_epsilon,
		"eval_seed_base": args.eval_seed_base,
		"replay_buffer_capacity": args.replay_buffer_capacity,
		"logdir": logdir,
		"checkpoint_dir": ckptdir,
		"update_freq": args.update_freq,
		"update_target_freq": args.update_target_freq,
		"learning_rate": args.learning_rate,
		"eval_interval": args.eval_interval,
		"eval_episode": args.eval_episode,
		"env_id": args.env,
		"variant": args.variant,
		"seed": args.seed,
		"num_envs": args.num_envs,
		"async_envs": not args.sync_envs,
		"clip_reward": args.clip_reward,
		"life_loss_terminal": args.life_loss_terminal,
		"loss": args.loss,
		"max_grad_norm": args.max_grad_norm,
		"log_interval": args.log_interval,
		"run_name": run_name,
		"resume": args.resume,
		"argv": " ".join(argv if argv is not None else sys.argv[1:]),
	}
	save_config(config, os.path.join(logdir, "config.json"))

	print(f"run_name: {run_name}")
	print(f"logdir:   {logdir}")
	print(f"ckptdir:  {ckptdir}")

	agent = AtariDQNAgent(config)
	print(f"device: {agent.device}  actions: {agent.num_actions}  "
	      f"replay buffer: {agent.replay_buffer.memory_bytes() / 1e9:.2f} GB")

	if args.resume:
		agent.load_checkpoint(args.resume)

	try:
		agent.train()
	except KeyboardInterrupt:
		print("interrupted -- saving latest.pt")
		agent.save_checkpoint("latest.pt")
	finally:
		agent.writer.flush()
		agent.writer.close()
		agent.env.close()
		agent.test_env.close()


if __name__ == "__main__":
	# Guard required on Windows: AsyncVectorEnv spawns subprocesses, which
	# re-import this module.
	main()
