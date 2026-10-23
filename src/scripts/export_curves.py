"""Export TensorBoard scalars to CSV + matplotlib PNG.

The report wants TensorBoard screenshots; these are the backup figures and the
raw numbers behind them.

    python scripts/export_curves.py                     # every run, every tag
    python scripts/export_curves.py --compare dqn_run ddqn_run --tag "Train/Q Mean"
    python scripts/export_curves.py --compare dqn_run par_run --x wall

Comparison plots overlay the runs on one axes with a legend; each curve is
drawn twice, faint at full resolution and solid after smoothing, so the figure
shows both the trend and the real variance.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
	"Train/Episode Reward",
	"Evaluate/Episode Reward",
	"Train/Episode Length",
	"Train/Loss",
	"Train/Q Mean",
	"Train/Epsilon",
	"Perf/Steps Per Second",
]


def load_run(run_dir):
	"""{tag: (steps, values, wall_times)} for one run directory."""
	acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
	acc.Reload()
	out = {}
	for tag in acc.Tags().get("scalars", []):
		events = acc.Scalars(tag)
		out[tag] = (
			np.array([e.step for e in events], dtype=np.int64),
			np.array([e.value for e in events], dtype=np.float64),
			np.array([e.wall_time for e in events], dtype=np.float64),
		)
	return out


def smooth(values, weight):
	"""TensorBoard's exponential moving average (same `weight` semantics)."""
	if weight <= 0 or len(values) == 0:
		return np.asarray(values, dtype=np.float64)
	out = np.empty(len(values), dtype=np.float64)
	acc = 0.0
	for i, v in enumerate(values):
		acc = acc * weight + (1 - weight) * float(v)
		# de-bias the first few points, which are otherwise pulled toward 0
		out[i] = acc / (1 - weight ** (i + 1))
	return out


def slug(text):
	return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def x_axis(steps, walls, mode):
	if mode == "wall":
		return (walls - walls[0]) / 3600.0, "wall-clock hours"
	return steps, "env steps"


def write_csv(path, steps, values, walls):
	with open(path, "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["env_step", "value", "wall_time"])
		w.writerows(zip(steps.tolist(), values.tolist(), walls.tolist()))


def plot_single(run_name, tag, data, out_dir, weight, x_mode):
	steps, values, walls = data
	if len(steps) == 0:
		return None
	xs, xlabel = x_axis(steps, walls, x_mode)
	fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
	ax.plot(xs, values, linewidth=0.8, alpha=0.25, color="tab:orange")
	ax.plot(xs, smooth(values, weight), linewidth=1.6, color="tab:orange")
	ax.set_xlabel(xlabel)
	ax.set_ylabel(tag)
	ax.set_title(f"{run_name}\n{tag}")
	ax.grid(alpha=0.3)
	fig.tight_layout()
	path = os.path.join(out_dir, f"{run_name}__{slug(tag)}.png")
	fig.savefig(path)
	plt.close(fig)
	return path


def plot_compare(runs, tag, out_dir, weight, x_mode, label=None):
	"""One axes, one line per run, legend included."""
	fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
	colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
	plotted = 0
	for k, (run_name, tags) in enumerate(runs):
		if tag not in tags:
			continue
		steps, values, walls = tags[tag]
		if len(steps) == 0:
			continue
		xs, xlabel = x_axis(steps, walls, x_mode)
		c = colors[k % len(colors)]
		ax.plot(xs, values, linewidth=0.7, alpha=0.20, color=c)
		ax.plot(xs, smooth(values, weight), linewidth=1.8, color=c, label=run_name)
		plotted += 1
	if not plotted:
		plt.close(fig)
		return None
	ax.set_xlabel(xlabel)
	ax.set_ylabel(tag)
	ax.set_title(tag + ("  (wall-clock)" if x_mode == "wall" else ""))
	ax.grid(alpha=0.3)
	ax.legend(fontsize=8)
	fig.tight_layout()
	name = label or "_vs_".join(r for r, _ in runs)
	path = os.path.join(out_dir, f"compare__{slug(name)}__{slug(tag)}"
	                             f"{'__wall' if x_mode == 'wall' else ''}.png")
	fig.savefig(path)
	plt.close(fig)
	return path


def main(argv=None):
	p = argparse.ArgumentParser(description="Export TensorBoard scalars")
	p.add_argument("--runs-dir", default=None)
	p.add_argument("--out-dir", default=None, help="default: <repo>/figures")
	p.add_argument("--runs", nargs="*", default=None, help="run names (default: all)")
	p.add_argument("--compare", nargs="*", default=None,
	               help="run names to overlay on one figure")
	p.add_argument("--tag", nargs="*", default=None, help="tags (default: the standard set)")
	p.add_argument("--smoothing", type=float, default=0.9, help="EMA weight, 0 disables")
	p.add_argument("--x", default="step", choices=("step", "wall"))
	p.add_argument("--label", default=None, help="filename label for a comparison")
	p.add_argument("--no-csv", action="store_true")
	args = p.parse_args(argv)

	repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	runs_dir = args.runs_dir or os.path.join(repo_root, "runs")
	out_dir = args.out_dir or os.path.join(repo_root, "figures")
	os.makedirs(out_dir, exist_ok=True)
	tags = args.tag or TAGS

	names = args.compare or args.runs or sorted(
		d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)))
	loaded = [(n, load_run(os.path.join(runs_dir, n))) for n in names]

	written = []
	if args.compare:
		for tag in tags:
			path = plot_compare(loaded, tag, out_dir, args.smoothing, args.x, args.label)
			if path:
				written.append(path)
	else:
		for run_name, run_tags in loaded:
			for tag in tags:
				if tag not in run_tags:
					continue
				path = plot_single(run_name, tag, run_tags[tag], out_dir,
				                   args.smoothing, args.x)
				if path:
					written.append(path)
				if not args.no_csv:
					csv_path = os.path.join(out_dir, f"{run_name}__{slug(tag)}.csv")
					write_csv(csv_path, *run_tags[tag])
					written.append(csv_path)

	for w in written:
		print(w)
	print(f"\n{len(written)} file(s) written to {out_dir}")


if __name__ == "__main__":
	main()
