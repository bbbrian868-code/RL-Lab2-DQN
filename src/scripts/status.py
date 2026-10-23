"""Monitor every run under `runs/`.

    python scripts/status.py            # one snapshot
    python scripts/status.py --watch 30 # refresh every 30 s

Reads only the files the training loop already writes (config.json, eval.csv,
train.log), so it is safe to run against live experiments.
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import fmt_eta

LINE_RE = re.compile(
	r"\[(?P<step>\d+)/(?P<total>\d+)\].*?sps:\s*(?P<sps>[\d.]+)")


def tail(path, nbytes=200_000):
	with open(path, "rb") as f:
		f.seek(0, os.SEEK_END)
		f.seek(max(0, f.tell() - nbytes))
		return f.read().decode("utf-8", errors="replace")


def read_run(run_dir):
	name = os.path.basename(run_dir)
	info = {"run": name, "step": 0, "total": 0, "sps": 0.0,
	        "eval_step": None, "eval_mean": None, "best": None, "age": None}

	cfg_path = os.path.join(run_dir, "config.json")
	if os.path.exists(cfg_path):
		with open(cfg_path, encoding="utf-8") as f:
			cfg = json.load(f)
		info["total"] = int(cfg.get("training_steps", 0))
		info["variant"] = cfg.get("variant", "?")
		info["env"] = str(cfg.get("env_id", "?")).split("/")[-1]
		info["num_envs"] = cfg.get("num_envs", 1)

	log_path = os.path.join(run_dir, "train.log")
	if os.path.exists(log_path):
		info["age"] = (dt.datetime.now()
		               - dt.datetime.fromtimestamp(os.path.getmtime(log_path))).total_seconds()
		text = tail(log_path)
		info["finished"] = "training finished at env step" in text
		for m in LINE_RE.finditer(text):
			info["step"] = int(m.group("step"))
			info["total"] = int(m.group("total")) or info["total"]
			info["sps"] = float(m.group("sps"))

	eval_path = os.path.join(run_dir, "eval.csv")
	if os.path.exists(eval_path):
		with open(eval_path, encoding="utf-8") as f:
			rows = list(csv.DictReader(f))
		if rows:
			info["eval_step"] = int(rows[-1]["env_step"])
			info["eval_mean"] = float(rows[-1]["mean"])
			info["best"] = max(float(r["mean"]) for r in rows)
	return info


def main(argv=None):
	p = argparse.ArgumentParser(description="Status of all training runs")
	p.add_argument("--runs-dir", default=None)
	p.add_argument("--watch", type=float, default=None, help="refresh interval in seconds")
	args = p.parse_args(argv)

	repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	runs_dir = args.runs_dir or os.path.join(repo_root, "runs")

	while True:
		if args.watch:
			os.system("cls" if os.name == "nt" else "clear")
		dirs = sorted(d for d in (os.path.join(runs_dir, x) for x in os.listdir(runs_dir))
		              if os.path.isdir(d)) if os.path.isdir(runs_dir) else []
		print(f"{'run':<30}{'env':<14}{'variant':<14}{'step':>12} {'%':>6} "
		      f"{'sps':>7} {'eta':>10} {'eval':>9} {'best':>9}  state")
		print("-" * 132)
		for d in dirs:
			i = read_run(d)
			pct = 100.0 * i["step"] / i["total"] if i["total"] else 0.0
			eta = (i["total"] - i["step"]) / i["sps"] if i["sps"] > 0 else None
			# No new log line for 5 minutes almost certainly means the process died.
			state = "running" if (i["age"] is not None and i["age"] < 300) else "stopped/idle"
			if i.get("finished"):
				state = "done"
			variant = i.get("variant", "?")
			if i.get("num_envs", 1) > 1:
				variant += f" x{i['num_envs']}"
			last_eval = "-" if i["eval_mean"] is None else f"{i['eval_mean']:.0f}"
			best_eval = "-" if i["best"] is None else f"{i['best']:.0f}"
			print(f"{i['run']:<30}{i.get('env', '?'):<14}{variant:<14}"
			      f"{i['step']:>12,} {pct:>5.1f}% {i['sps']:>7.1f} {fmt_eta(eta):>10} "
			      f"{last_eval:>9} {best_eval:>9}  {state}")
		if not dirs:
			print("(no runs yet)")
		if not args.watch:
			return
		import time
		time.sleep(args.watch)


if __name__ == "__main__":
	main()
