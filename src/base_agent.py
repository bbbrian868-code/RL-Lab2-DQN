"""DQNBaseAgent -- the TA sample skeleton, filled in and extended.

Kept from the sample: the class name, the abstract `decide_agent_actions` /
`update_behavior_network` pair that subclasses override, `update`,
`update_target_network`, `epsilon_decay`, `train`, `evaluate`, `save`, `load`
and `load_and_evaluate`.

Added for this lab:
  * env-step-based (not episode-based) evaluation and checkpointing;
  * raw-score logging alongside clipped-reward training;
  * resumable checkpoints (nets, optimizer, step counters, RNG state);
  * a parallelized-rollout training loop sharing all of the above;
  * the fixed TensorBoard tag set the report comparisons are built on.

The sample's two `### TODO ###` blocks here sit in abstract methods; they are
implemented in dqn_agent_atari.py as [TODO 3] and [TODO 4].  The
parallel-rollout loop is marked [EXTRA: parallelized rollout].
"""

import csv
import os
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from atari_wrappers import make_atari_env, make_vector_env
from replay_buffer.replay_buffer import ReplayMemory
from utils import fmt_eta, rng_state, set_rng_state


class DQNBaseAgent(ABC):
	def __init__(self, config):
		self.config = dict(config)
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.epsilon = 1.0
		self.eps_min = config["eps_min"]
		self.eps_decay = int(config["eps_decay"])
		self.eval_epsilon = config["eval_epsilon"]
		self.warmup_steps = int(config["warmup_steps"])
		self.eval_interval = int(config["eval_interval"])
		self.eval_episode = int(config["eval_episode"])
		self.gamma = config["gamma"]
		self.update_freq = int(config["update_freq"])
		self.update_target_freq = int(config["update_target_freq"])

		self.env_id = config["env_id"]
		self.variant = config["variant"]
		self.seed = int(config["seed"])
		self.num_envs = int(config.get("num_envs", 1))
		self.clip_reward = bool(config.get("clip_reward", True))
		self.life_loss_terminal = bool(config.get("life_loss_terminal", False))
		self.max_grad_norm = config.get("max_grad_norm", 10.0)
		self.log_interval = int(config.get("log_interval", 100))   # updates per scalar flush
		self.run_name = config["run_name"]
		self.checkpoint_dir = config["checkpoint_dir"]
		os.makedirs(self.checkpoint_dir, exist_ok=True)

		self.replay_buffer = ReplayMemory(
			int(config["replay_buffer_capacity"]),
			num_streams=self.num_envs,
			seed=self.seed,
		)
		self.writer = SummaryWriter(config["logdir"])
		self.eval_csv = os.path.join(config["logdir"], "eval.csv")

		# --- bookkeeping -------------------------------------------------
		self._updates_done = 0
		self._last_target_sync = 0
		# Gradient steps start here.  Separate from `warmup_steps` (which fixes
		# the epsilon schedule) because a resumed run has to re-fill its replay
		# buffer without rewinding epsilon back to 1.0.
		self._update_start_step = self.warmup_steps
		self._next_eval_at = self.warmup_steps + self.eval_interval
		self.best_eval_score = -float("inf")
		self.episode_idx = 0
		self._loss_acc, self._q_acc, self._acc_n = 0.0, 0.0, 0
		self._perf_t0, self._perf_step0 = None, None
		self._elapsed_before_resume = 0.0
		self._session_t0 = time.time()
		self._resumed = False

	# ================================================================ core

	@abstractmethod
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		"""Epsilon-greedy action(s) from the behavior network.

		[TODO 3] -- implemented in AtariDQNAgent.decide_agent_actions
		(dqn_agent_atari.py).
		"""

	def update(self):
		"""Keep one gradient step per `update_freq` env steps, and sync the
		target net on the env-step grid.

		Written against counters rather than `step % freq == 0` so that the
		ratio is preserved when the parallel rollout advances the step count by
		`num_envs` at a time.
		"""
		owed = max(0, (self.total_time_step - self._update_start_step)) // self.update_freq
		while self._updates_done < owed:
			self.update_behavior_network()
			self._updates_done += 1

		sync_epoch = self.total_time_step // self.update_target_freq
		if sync_epoch > self._last_target_sync:
			self.update_target_network()
			self._last_target_sync = sync_epoch

	@abstractmethod
	def update_behavior_network(self):
		"""One gradient step on a sampled minibatch.

		[TODO 4] -- implemented in AtariDQNAgent.update_behavior_network
		(dqn_agent_atari.py).
		"""

	def update_target_network(self):
		self.target_net.load_state_dict(self.behavior_net.state_dict())

	def _epsilon_at(self, step):
		"""Linear 1.0 -> eps_min over `eps_decay` env steps after warmup.

		Computed from the step counter instead of being decremented in place,
		so resuming from a checkpoint lands on exactly the right value.
		"""
		if step < self.warmup_steps:
			return 1.0
		frac = (step - self.warmup_steps) / max(1, self.eps_decay)
		return max(self.eps_min, 1.0 - frac * (1.0 - self.eps_min))

	def epsilon_decay(self):
		self.epsilon = self._epsilon_at(self.total_time_step)

	# ============================================================ training

	def train(self):
		self._perf_t0 = time.time()
		self._perf_step0 = self.total_time_step
		if self.num_envs > 1:
			self._train_vector()
		else:
			self._train_single()
		self.save_checkpoint("latest.pt")
		print(f"training finished at env step {self.total_time_step}")

	def _train_single(self):
		env = self.env
		observation, info = env.reset(seed=self.seed)
		episode_start = True
		episode_reward, episode_len = 0.0, 0
		lives = info.get("lives", None)
		ep_t0 = time.time()

		while self.total_time_step < self.training_steps:
			self.epsilon_decay()   # epsilon is 1.0 for the whole warmup period
			action = self.decide_agent_actions(observation, self.epsilon, env.action_space)

			next_observation, reward, terminated, truncated, info = env.step(action)
			done = bool(terminated or truncated)
			episode_reward += float(reward)   # raw game score, for the report
			episode_len += 1

			# Losing a life can be treated as terminal for the TD target only;
			# the env is NOT reset, so the episode (and the game score) runs on.
			new_lives = info.get("lives", lives)
			life_lost = lives is not None and new_lives is not None and new_lives < lives
			lives = new_lives
			bootstrap_terminal = bool(terminated or (self.life_loss_terminal and life_lost))

			r = float(np.clip(reward, -1.0, 1.0)) if self.clip_reward else float(reward)
			self.replay_buffer.append(
				frame=observation[-1], action=int(action), reward=r,
				terminated=bootstrap_terminal, episode_done=done,
				episode_start=episode_start,
				next_frame=next_observation[-1] if done else None,
			)
			episode_start = False
			self.total_time_step += 1

			if self.total_time_step >= self._update_start_step:
				self.update()

			if done:
				self.episode_idx += 1
				self._log_episode(episode_reward, episode_len, time.time() - ep_t0)
				observation, info = env.reset()
				lives = info.get("lives", None)
				episode_start = True
				episode_reward, episode_len = 0.0, 0
				ep_t0 = time.time()
			else:
				observation = next_observation

			self._maybe_eval()

	def _train_vector(self):
		"""[EXTRA: parallelized rollout] `num_envs` envs stepped together.

		The step counter advances by `num_envs` per vector step, and warmup,
		epsilon decay, target sync, evaluation and the update ratio are all
		driven by that *total* count -- so a run with 8 envs sees the same
		schedule, in env steps, as a single-env run.

		Autoreset is NEXT_STEP: the step after an episode ends is a filler that
		only delivers the reset observation, and is not stored.
		"""
		venv = self.env
		observations, infos = venv.reset(seed=[self.seed + i for i in range(self.num_envs)])
		n = self.num_envs
		episode_start = np.ones(n, dtype=bool)
		skip_next = np.zeros(n, dtype=bool)     # autoreset filler step
		ep_reward = np.zeros(n, dtype=np.float64)
		ep_len = np.zeros(n, dtype=np.int64)
		ep_t0 = np.full(n, time.time())
		lives = self._vec_lives(infos, n)

		while self.total_time_step < self.training_steps:
			self.epsilon_decay()   # epsilon is 1.0 for the whole warmup period
			actions = self.decide_agent_actions(observations, self.epsilon,
			                                    venv.single_action_space)

			next_observations, rewards, terminateds, truncateds, infos = venv.step(actions)
			new_lives = self._vec_lives(infos, n)
			dones = np.logical_or(terminateds, truncateds)

			for i in range(n):
				if skip_next[i]:
					# This transition is the autoreset filler; discard it and
					# treat next_observations[i] as a fresh episode start.
					continue
				ep_reward[i] += float(rewards[i])
				ep_len[i] += 1
				life_lost = (lives is not None and new_lives is not None
				             and new_lives[i] < lives[i])
				bootstrap_terminal = bool(terminateds[i] or (self.life_loss_terminal and life_lost))
				r = float(np.clip(rewards[i], -1.0, 1.0)) if self.clip_reward else float(rewards[i])
				self.replay_buffer.append(
					frame=observations[i][-1], action=int(actions[i]), reward=r,
					terminated=bootstrap_terminal, episode_done=bool(dones[i]),
					episode_start=bool(episode_start[i]),
					next_frame=next_observations[i][-1] if dones[i] else None,
					stream=i,
				)
				episode_start[i] = False
				self.total_time_step += 1

				if dones[i]:
					self.episode_idx += 1
					self._log_episode(ep_reward[i], int(ep_len[i]), time.time() - ep_t0[i])
					ep_reward[i], ep_len[i] = 0.0, 0
					ep_t0[i] = time.time()

			# After a filler step the lane restarts; after a done the *next*
			# step is the filler.
			episode_start = np.where(skip_next, True, episode_start)
			skip_next = dones.copy()
			lives = new_lives
			observations = next_observations

			if self.total_time_step >= self._update_start_step:
				self.update()
			self._maybe_eval()

	@staticmethod
	def _vec_lives(infos, n):
		lv = infos.get("lives", None)
		if lv is None:
			return None
		return np.asarray(lv).reshape(-1)[:n]

	# ============================================================= logging

	def _log_episode(self, episode_reward, episode_len, episode_wall):
		step = self.total_time_step
		sps = self._steps_per_second()
		self.writer.add_scalar("Train/Episode Reward", episode_reward, step)
		self.writer.add_scalar("Train/Episode Length", episode_len, step)
		self.writer.add_scalar("Train/Epsilon", self.epsilon, step)
		self.writer.add_scalar("Perf/Steps Per Second", sps, step)
		remaining = (self.training_steps - step) / sps if sps > 0 else None
		print(f"[{step}/{self.training_steps}] episode: {self.episode_idx}  "
		      f"episode reward: {episode_reward:.1f}  episode len: {episode_len}  "
		      f"epsilon: {self.epsilon:.4f}  sps: {sps:.1f}  eta: {fmt_eta(remaining)}")

	def _steps_per_second(self):
		now = time.time()
		dt = now - self._perf_t0
		if dt <= 0:
			return 0.0
		sps = (self.total_time_step - self._perf_step0) / dt
		if dt > 60:  # restart the window so the number tracks recent throughput
			self._perf_t0, self._perf_step0 = now, self.total_time_step
		return sps

	def _accumulate_train_scalars(self, loss, q_mean):
		self._loss_acc += float(loss)
		self._q_acc += float(q_mean)
		self._acc_n += 1
		if self._acc_n >= self.log_interval:
			self.writer.add_scalar("Train/Loss", self._loss_acc / self._acc_n, self.total_time_step)
			self.writer.add_scalar("Train/Q Mean", self._q_acc / self._acc_n, self.total_time_step)
			self._loss_acc, self._q_acc, self._acc_n = 0.0, 0.0, 0

	# ========================================================== evaluation

	def _maybe_eval(self):
		if self.total_time_step < self._next_eval_at:
			return
		while self._next_eval_at <= self.total_time_step:
			self._next_eval_at += self.eval_interval
		scores = self.evaluate()
		mean = float(np.mean(scores))
		self.writer.add_scalar("Evaluate/Episode Reward", mean, self.total_time_step)
		self._write_eval_csv(scores)
		self.save_checkpoint("latest.pt")
		if mean > self.best_eval_score:
			self.best_eval_score = mean
			self.save_checkpoint("best.pt")
			print(f"new best evaluation score {mean:.1f} -> best.pt")

	def evaluate(self):
		"""`eval_episode` full games on a separate env with a fixed seed list."""
		print("==============================================")
		print(f"Evaluating at env step {self.total_time_step} ...")
		self.behavior_net.eval()
		all_rewards = []
		for i in range(self.eval_episode):
			seed = self.config.get("eval_seed_base", 10000) + i
			observation, info = self.test_env.reset(seed=seed)
			total_reward = 0.0
			while True:
				action = self.decide_agent_actions(observation, self.eval_epsilon,
				                                   self.test_env.action_space)
				observation, reward, terminate, truncate, info = self.test_env.step(action)
				total_reward += float(reward)
				if terminate or truncate:
					print(f"episode {i + 1} (seed {seed}) reward: {total_reward}")
					all_rewards.append(total_reward)
					break
		avg = sum(all_rewards) / len(all_rewards)
		print(f"average score: {avg}")
		print("==============================================")
		self.behavior_net.train()
		return all_rewards

	def _write_eval_csv(self, scores):
		new = not os.path.exists(self.eval_csv)
		with open(self.eval_csv, "a", newline="", encoding="utf-8") as f:
			w = csv.writer(f)
			if new:
				w.writerow(["env_step", "wall_time", "mean", "std", "min", "max"]
				           + [f"ep{i + 1}" for i in range(self.eval_episode)])
			w.writerow([self.total_time_step, f"{self._wall_time():.1f}",
			            f"{np.mean(scores):.2f}", f"{np.std(scores):.2f}",
			            f"{np.min(scores):.2f}", f"{np.max(scores):.2f}"]
			           + [f"{s:.2f}" for s in scores])

	def _wall_time(self):
		return self._elapsed_before_resume + (time.time() - self._session_t0)

	# ========================================================= checkpoints

	def save(self, save_path):
		"""Sample-code behaviour: weights only."""
		torch.save(self.behavior_net.state_dict(), save_path)

	def load(self, load_path):
		"""Accepts both a bare state_dict and a full training checkpoint."""
		blob = torch.load(load_path, map_location=self.device, weights_only=False)
		state = blob.get("behavior_net", blob) if isinstance(blob, dict) else blob
		self.behavior_net.load_state_dict(state)
		return blob if isinstance(blob, dict) and "behavior_net" in blob else None

	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()

	def save_checkpoint(self, name):
		path = os.path.join(self.checkpoint_dir, name)
		tmp = path + ".tmp"
		torch.save({
			"behavior_net": self.behavior_net.state_dict(),
			"target_net": self.target_net.state_dict(),
			"optimizer": self.optim.state_dict(),
			"total_time_step": self.total_time_step,
			"epsilon": self.epsilon,
			"updates_done": self._updates_done,
			"last_target_sync": self._last_target_sync,
			"next_eval_at": self._next_eval_at,
			"best_eval_score": self.best_eval_score,
			"episode_idx": self.episode_idx,
			"elapsed": self._wall_time(),
			"config": self.config,
			"rng": rng_state(),
		}, tmp)
		os.replace(tmp, path)   # atomic: a crash mid-save cannot corrupt latest.pt

	def load_checkpoint(self, path):
		blob = torch.load(path, map_location=self.device, weights_only=False)
		self.behavior_net.load_state_dict(blob["behavior_net"])
		self.target_net.load_state_dict(blob["target_net"])
		self.optim.load_state_dict(blob["optimizer"])
		self.total_time_step = int(blob["total_time_step"])
		self.epsilon = float(blob["epsilon"])
		self._updates_done = int(blob["updates_done"])
		self._last_target_sync = int(blob["last_target_sync"])
		self._next_eval_at = int(blob["next_eval_at"])
		self.best_eval_score = float(blob["best_eval_score"])
		self.episode_idx = int(blob["episode_idx"])
		self._elapsed_before_resume = float(blob.get("elapsed", 0.0))
		set_rng_state(blob.get("rng"))
		self._resumed = True
		# The replay buffer is deliberately not checkpointed (it would be a
		# ~700 MB file per save).  Re-warm it before resuming updates -- but
		# collect that data with the *current* epsilon, not epsilon=1.0, so the
		# refilled buffer matches the policy we are resuming.
		self._update_start_step = self.total_time_step + int(
			self.config.get("resume_warmup_steps", 20000))
		self._updates_done = 0
		print(f"RESUMED from {path} at env step {self.total_time_step}; "
		      f"replay buffer starts empty, gradient steps resume at "
		      f"{self._update_start_step} (epsilon stays at {self.epsilon:.4f})")
		return blob
