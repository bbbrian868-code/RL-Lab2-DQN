"""Scan seeds with a trained checkpoint to find a high-scoring one for the demo.

    python find_best_seed.py --checkpoint ../checkpoints/<run>/best.pt --max-seed 50

Writes every seed's score to CSV and prints the top ones.  Pass the winning
seed to `test.py --seeds <s> --episodes 1 --record-video` for the demo; the
score is reproducible because the ALE RNG (including sticky actions) and the
epsilon-greedy coin flips are both seeded.
"""

import argparse
import csv
import datetime as dt
import os
import random

import numpy as np
import torch

torch.set_num_threads(1)

from atari_wrappers import make_atari_env
from test import load_agent_net, play_episode


def build_parser():
	p = argparse.ArgumentParser(description="Find the best demo seed for a checkpoint")
	p.add_argument("--checkpoint", required=True)
	p.add_argument("--env", default=None)
	p.add_argument("--variant", default=None)
	p.add_argument("--min-seed", type=int, default=0)
	p.add_argument("--max-seed", type=int, default=50)
	p.add_argument("--epsilon", type=float, default=0.01)
	p.add_argument("--repeats", type=int, default=1, help="games per seed (averaged)")
	p.add_argument("--out", default=None)
	p.add_argument("--cpu", action="store_true")
	return p


def main(argv=None):
	args = build_parser().parse_args(argv)
	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	device = torch.device("cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu")

	net, env_id, variant, cfg = load_agent_net(args.checkpoint, args.variant, args.env, device)
	run_name = cfg.get("run_name") or os.path.basename(
		os.path.dirname(os.path.abspath(args.checkpoint)))
	stamp = dt.datetime.now().strftime("%m%d-%H%M%S")
	out = args.out or os.path.join(repo_root, "test_logs",
	                               f"seedscan_{run_name}_{stamp}.csv")
	os.makedirs(os.path.dirname(out), exist_ok=True)

	env = make_atari_env(env_id)
	rows = []
	print(f"scanning seeds {args.min_seed}..{args.max_seed} on {env_id} ({variant})")
	for seed in range(args.min_seed, args.max_seed + 1):
		scores, lengths = [], []
		for r in range(args.repeats):
			# play_episode derives everything from `seed`, so this is exactly
			# the game `test.py --seeds <seed>` will replay.
			s, l = play_episode(net, env, seed, args.epsilon, device)
			scores.append(s)
			lengths.append(l)
		row = {"seed": seed, "mean_score": float(np.mean(scores)),
		       "max_score": float(np.max(scores)), "mean_length": float(np.mean(lengths))}
		rows.append(row)
		print(f"seed {seed:<4} mean {row['mean_score']:>8.1f}  max {row['max_score']:>8.1f}  "
		      f"len {row['mean_length']:>8.0f}")
	env.close()

	with open(out, "w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=["seed", "mean_score", "max_score", "mean_length"])
		w.writeheader()
		w.writerows(rows)

	rows.sort(key=lambda r: r["mean_score"], reverse=True)
	print("\ntop 10 seeds:")
	for r in rows[:10]:
		print(f"  seed {r['seed']:<4} mean {r['mean_score']:.1f}  max {r['max_score']:.1f}")
	best = rows[0]
	print(f"\nbest seed: {best['seed']} (score {best['mean_score']:.1f})")
	print(f"saved: {out}")
	print(f"\ndemo command:\n  python test.py --checkpoint {args.checkpoint} "
	      f"--seeds {best['seed']} --episodes 1 --record-video")
	return best


if __name__ == "__main__":
	main()
