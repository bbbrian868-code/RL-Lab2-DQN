"""Evaluate a trained checkpoint and print the score table used in the report.

    python test.py --checkpoint ../checkpoints/<run>/best.pt --episodes 5

Output matches the format in the lab handout:

    Episode: 1      Length: 13301   Total reward: 1040.00
    ...
    average score: 1037.6

Every run is also written to `test_logs/<run_name>_<timestamp>.txt` (the exact
text above, for screenshotting) and `.json` (machine-readable, including the
per-episode seeds so any result can be reproduced).
"""

import argparse
import datetime as dt
import json
import os
import random
import sys

import numpy as np
import torch

torch.set_num_threads(1)

from atari_wrappers import make_atari_env
from models.atari_model import build_network
from utils import Tee

VARIANTS = ("dqn", "ddqn", "dueling", "dueling_ddqn")


def build_parser():
	p = argparse.ArgumentParser(description="Test a DQN checkpoint")
	p.add_argument("--checkpoint", required=True)
	p.add_argument("--env", default=None, help="default: the env the checkpoint was trained on")
	p.add_argument("--variant", default=None, help="default: read from the checkpoint")
	p.add_argument("--episodes", type=int, default=5)
	p.add_argument("--seed", type=int, default=0, help="first episode seed; episode i uses seed+i")
	p.add_argument("--seeds", type=int, nargs="*", default=None,
	               help="explicit per-episode seeds, overrides --seed")
	p.add_argument("--epsilon", type=float, default=0.01)
	p.add_argument("--record-video", action="store_true")
	p.add_argument("--render", action="store_true", help="open a live window (needs a display)")
	p.add_argument("--video-dir", default=None)
	p.add_argument("--out-dir", default=None, help="default: <repo>/test_logs")
	p.add_argument("--no-log", action="store_true", help="print only, do not write log files")
	p.add_argument("--cpu", action="store_true")
	return p


def load_agent_net(checkpoint, variant, env_id, device):
	blob = torch.load(checkpoint, map_location=device, weights_only=False)
	cfg = blob.get("config", {}) if isinstance(blob, dict) else {}
	env_id = env_id or cfg.get("env_id", "ALE/MsPacman-v5")
	variant = variant or cfg.get("variant", "dqn")
	assert variant in VARIANTS, f"unknown variant {variant}"

	probe = make_atari_env(env_id)
	num_actions = probe.action_space.n
	probe.close()

	net = build_network(variant, num_actions).to(device)
	state = blob.get("behavior_net", blob) if isinstance(blob, dict) else blob
	net.load_state_dict(state)
	net.eval()
	return net, env_id, variant, cfg


def play_episode(net, env, seed, epsilon, device):
	"""One full game, fully determined by `seed`.

	Three things have to be seeded for a replay to be bit-identical: the ALE
	itself (including sticky actions and the random no-ops at reset), which
	`reset(seed=...)` covers; the action space sampler; and the epsilon-greedy
	coin flips.  The last two are derived from the same episode seed here, so
	`--seeds <s>` reproduces exactly what the seed scan measured for `<s>`,
	independently of how many episodes ran before it.
	"""
	rng = random.Random(seed)
	num_actions = env.action_space.n
	observation, _ = env.reset(seed=int(seed))
	total_reward, length = 0.0, 0
	while True:
		if rng.random() < epsilon:
			action = rng.randrange(num_actions)
		else:
			with torch.no_grad():
				s = torch.as_tensor(np.asarray(observation), dtype=torch.uint8,
				                    device=device).unsqueeze(0)
				action = int(net(s).argmax(dim=1).item())
		observation, reward, terminated, truncated, _ = env.step(action)
		total_reward += float(reward)
		length += 1
		if terminated or truncated:
			return total_reward, length


def main(argv=None):
	args = build_parser().parse_args(argv)
	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	device = torch.device("cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu")

	net, env_id, variant, cfg = load_agent_net(args.checkpoint, args.variant, args.env, device)
	run_name = cfg.get("run_name") or os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
	stamp = dt.datetime.now().strftime("%m%d-%H%M%S")

	if not args.no_log:
		out_dir = args.out_dir or os.path.join(repo_root, "test_logs")
		os.makedirs(out_dir, exist_ok=True)
		txt_path = os.path.join(out_dir, f"{run_name}_{stamp}.txt")
		json_path = os.path.join(out_dir, f"{run_name}_{stamp}.json")
		sys.stdout = Tee(txt_path, sys.__stdout__)
	else:
		txt_path = json_path = None

	seeds = args.seeds if args.seeds else [args.seed + i for i in range(args.episodes)]
	episodes = len(seeds)

	render_mode = "rgb_array" if args.record_video else ("human" if args.render else None)
	env = make_atari_env(env_id, render_mode=render_mode)
	video_dir = None
	if args.record_video:
		from gymnasium.wrappers import RecordVideo
		video_dir = args.video_dir or os.path.join(repo_root, "videos", run_name)
		os.makedirs(video_dir, exist_ok=True)
		env = RecordVideo(env, video_folder=video_dir, name_prefix=f"{run_name}_{stamp}",
		                  episode_trigger=lambda _ep: True, disable_logger=True)

	print(f"checkpoint: {args.checkpoint}")
	print(f"env: {env_id}    variant: {variant}    epsilon: {args.epsilon}    device: {device}")
	print(f"seeds: {seeds}")
	print("")

	scores, lengths = [], []
	for i, seed in enumerate(seeds):
		reward, length = play_episode(net, env, seed, args.epsilon, device)
		scores.append(reward)
		lengths.append(length)
		print(f"Episode: {i + 1:<6}Length: {length:<10}Total reward: {reward:.2f}")

	avg = sum(scores) / episodes
	# `round` (not an f-string precision) so a clean mean prints as the handout
	# shows it -- "average score: 1037.6", not "1037.60" or "1037.5999999".
	print(f"average score: {round(avg, 2)}")
	env.close()

	if json_path:
		with open(json_path, "w", encoding="utf-8") as f:
			json.dump({
				"checkpoint": os.path.abspath(args.checkpoint),
				"run_name": run_name,
				"env": env_id,
				"variant": variant,
				"epsilon": args.epsilon,
				"episodes": episodes,
				"seeds": list(seeds),
				"scores": scores,
				"lengths": lengths,
				"average_score": avg,
				"best_score": max(scores),
				"best_seed": seeds[int(np.argmax(scores))],
				"video_dir": video_dir,
				"tested_at": dt.datetime.now().isoformat(timespec="seconds"),
			}, f, indent=2)
		print(f"\nsaved: {txt_path}")
		print(f"saved: {json_path}")
	if video_dir:
		print(f"videos: {video_dir}")
	return avg


if __name__ == "__main__":
	main()
