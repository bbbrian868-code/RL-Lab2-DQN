"""Small shared helpers: run naming, stdout teeing, config dumping."""

import datetime as _dt
import json
import os
import random
import sys

import numpy as np


def make_run_name(env_id, variant, seed, timestamp=None):
	"""`<env>_<variant>_s<seed>_<MMDD-HHMM>`, e.g. MsPacman_ddqn_s0_1010-0300."""
	env_short = env_id.split("/")[-1].split("-")[0]
	ts = timestamp or _dt.datetime.now().strftime("%m%d-%H%M")
	return f"{env_short}_{variant}_s{seed}_{ts}"


class Tee:
	"""Duplicate stdout/stderr into a timestamped log file."""

	def __init__(self, path, stream):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		self.file = open(path, "a", encoding="utf-8", buffering=1)
		self.stream = stream
		self._at_line_start = True

	def write(self, data):
		self.stream.write(data)
		for i, part in enumerate(data.split("\n")):
			if i:
				self.file.write("\n")
				self._at_line_start = True
			if not part:
				continue
			if self._at_line_start:
				self.file.write(_dt.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] "))
				self._at_line_start = False
			self.file.write(part)

	def flush(self):
		self.stream.flush()
		self.file.flush()

	def isatty(self):
		return getattr(self.stream, "isatty", lambda: False)()


def tee_stdout(path):
	sys.stdout = Tee(path, sys.__stdout__)
	sys.stderr = Tee(path, sys.__stderr__)


def save_config(config, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(config, f, indent=2, sort_keys=True, default=str)


def seed_everything(seed):
	import torch
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def rng_state():
	import torch
	return {
		"python": random.getstate(),
		"numpy": np.random.get_state(),
		"torch": torch.get_rng_state(),
		"torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
	}


def set_rng_state(state):
	import torch
	if state is None:
		return
	random.setstate(state["python"])
	np.random.set_state(state["numpy"])

	# torch.load(map_location="cuda") moves these to the GPU, but the RNG
	# setters insist on a CPU ByteTensor.
	as_byte = lambda t: t.cpu().to(torch.uint8)

	torch.set_rng_state(as_byte(state["torch"]))
	if state.get("torch_cuda") is not None and torch.cuda.is_available():
		try:
			torch.cuda.set_rng_state_all([as_byte(t) for t in state["torch_cuda"]])
		except (RuntimeError, ValueError, TypeError):
			pass


def fmt_eta(seconds):
	if seconds is None or seconds != seconds or seconds < 0:
		return "?"
	seconds = int(seconds)
	d, rem = divmod(seconds, 86400)
	h, rem = divmod(rem, 3600)
	m, s = divmod(rem, 60)
	if d:
		return f"{d}d{h:02d}h{m:02d}m"
	if h:
		return f"{h}h{m:02d}m"
	return f"{m}m{s:02d}s"
