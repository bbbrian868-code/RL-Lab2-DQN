"""Frame-efficient uint8 replay memory for Atari DQN.

Extends the TA sample `ReplayMemory` (same class name / `append` / `sample` /
`__len__` surface) with two changes that are required to keep 100k transitions
inside the RAM budget:

1.  Observations are kept as **uint8** and only converted to float (and divided
    by 255) inside the network's ``forward``.  Nothing float-sized is ever
    stored.
2.  Only the **newest 84x84 frame** of each stacked observation is stored.  The
    4-frame stack is rebuilt at sample time from neighbouring indices, so each
    frame is stored once instead of four times.

    100000 * 84 * 84 bytes = 706 MB for the frames, plus ~2 MB of metadata.
    Storing full (4,84,84) states *and* next-states would cost ~5.6 GB.

Episode boundaries
------------------
``first[i]`` marks the reset observation of an episode.  When the walk-back that
rebuilds a stack reaches a ``first`` frame it repeats that frame for the
remaining (older) slots.  This reproduces exactly what
``gymnasium.wrappers.FrameStackObservation`` does on reset (``padding_type
="reset"``), so the states the agent trains on are identical to the states it
acted on.

To make ``next_state`` always reconstructible, the terminal observation of an
episode is appended as a **pad entry** (``is_pad=True``).  Pad entries are never
themselves sampled, but they supply the final frame for the last real
transition of the episode.

terminated vs. truncated
------------------------
``terminated`` is stored separately from the episode boundary.  A *truncated*
episode (time limit) ends the episode but stores ``terminated=False``, so the TD
target still bootstraps through it -- which is what the Bellman equation calls
for, because the state was not actually absorbing.

Parallel rollout
----------------
With ``num_streams > 1`` the buffer holds one independent circular region per
vector-env lane.  Interleaving several envs into a single circular region would
corrupt the index-based stack reconstruction, because consecutive slots would
come from different episodes.
"""

import numpy as np
import torch


class ReplayMemory(object):
	def __init__(self, capacity, frame_stack=4, frame_shape=(84, 84), num_streams=1, seed=None):
		assert capacity >= num_streams * (frame_stack + 2)
		self.capacity = int(capacity)
		self.stack = int(frame_stack)
		self.num_streams = int(num_streams)
		# per-stream circular region
		self.cap = self.capacity // self.num_streams
		self.frame_shape = tuple(frame_shape)

		shape = (self.num_streams, self.cap)
		self.frames = np.zeros(shape + self.frame_shape, dtype=np.uint8)
		self.actions = np.zeros(shape, dtype=np.int64)
		self.rewards = np.zeros(shape, dtype=np.float32)
		self.terminated = np.zeros(shape, dtype=bool)   # bootstrap mask
		self.first = np.zeros(shape, dtype=bool)        # episode-start marker
		self.is_pad = np.zeros(shape, dtype=bool)       # terminal-frame filler

		self.pos = np.zeros(self.num_streams, dtype=np.int64)
		self.full = np.zeros(self.num_streams, dtype=bool)
		self.rng = np.random.default_rng(seed)

	# ------------------------------------------------------------------ write

	def __len__(self):
		return int(np.sum(np.where(self.full, self.cap, self.pos)))

	def _push(self, stream, frame, action, reward, terminated, first, is_pad):
		p = self.pos[stream]
		self.frames[stream, p] = frame
		self.actions[stream, p] = action
		self.rewards[stream, p] = reward
		self.terminated[stream, p] = terminated
		self.first[stream, p] = first
		self.is_pad[stream, p] = is_pad
		self.pos[stream] = (p + 1) % self.cap
		if self.pos[stream] == 0:
			self.full[stream] = True

	def append(self, frame, action, reward, terminated, episode_done, episode_start,
	           next_frame=None, stream=0):
		"""Store one transition.

		Args:
			frame:          newest (84,84) uint8 frame of the observation s_t.
			action:         int action taken at s_t.
			reward:         reward (already clipped, if clipping is enabled).
			terminated:     True only for a real absorbing state -> no bootstrap.
			                With ``--life-loss-terminal`` this is also set on a
			                lost life, which affects the TD target only.
			episode_done:   terminated or truncated -- the episode ended here.
			episode_start:  s_t is the reset observation of an episode.
			next_frame:     newest frame of s_{t+1}.  Required when
			                ``episode_done`` is True so the terminal observation
			                can be stored as a pad entry.
			stream:         vector-env lane index.
		"""
		self._push(stream, frame, action, reward, bool(terminated), bool(episode_start), False)
		if episode_done:
			assert next_frame is not None, "next_frame is required when the episode ends"
			# Pad entry: supplies next_state for the transition above and is
			# never sampled itself.
			self._push(stream, next_frame, 0, 0.0, False, False, True)

	# ----------------------------------------------------------------- sample

	def _stack_indices(self, stream, idx):
		"""Indices of the `stack` frames ending at `idx`, repeating on episode start."""
		out = np.empty((len(idx), self.stack), dtype=np.int64)
		out[:, -1] = idx
		cur = idx.copy()
		stop = np.zeros(len(idx), dtype=bool)
		for k in range(self.stack - 2, -1, -1):
			stop |= self.first[stream, cur]
			cur = np.where(stop, cur, (cur - 1) % self.cap)
			out[:, k] = cur
		return out

	def _valid(self, stream, idx):
		newest = (self.pos[stream] - 1) % self.cap
		ok = ~self.is_pad[stream, idx]
		ok &= idx != newest                       # need idx+1 to exist
		if bool(self.full[stream]):
			# Do not walk back into slots that have already been overwritten.
			ok &= ((idx - self.pos[stream]) % self.cap) >= (self.stack - 1)
		return ok

	def _sample_indices(self, batch_size):
		streams = self.rng.integers(0, self.num_streams, size=batch_size)
		out_s = np.empty(batch_size, dtype=np.int64)
		out_i = np.empty(batch_size, dtype=np.int64)
		filled = 0
		for _ in range(200):
			need = batch_size - filled
			if need == 0:
				break
			s = streams[filled:filled + need]
			n = np.where(self.full[s], self.cap, self.pos[s])
			cand = (self.rng.random(need) * n).astype(np.int64)
			keep = np.zeros(need, dtype=bool)
			for st in np.unique(s):
				m = s == st
				keep[m] = self._valid(st, cand[m])
			k = int(keep.sum())
			out_s[filled:filled + k] = s[keep]
			out_i[filled:filled + k] = cand[keep]
			filled += k
		if filled < batch_size:
			raise RuntimeError("replay buffer could not provide enough valid samples")
		return out_s, out_i

	def sample(self, batch_size, device):
		"""Returns (state, action, reward, next_state, terminated) as torch tensors.

		States come back as **uint8** (B, stack, 84, 84); the network casts and
		scales them.  This keeps the host->device copy 4x smaller than float32.
		"""
		s, i = self._sample_indices(batch_size)
		cur_idx = np.empty((batch_size, self.stack), dtype=np.int64)
		nxt_idx = np.empty((batch_size, self.stack), dtype=np.int64)
		for st in np.unique(s):
			m = s == st
			cur_idx[m] = self._stack_indices(st, i[m])
			nxt_idx[m] = self._stack_indices(st, (i[m] + 1) % self.cap)

		rows = s[:, None]
		state = self.frames[rows, cur_idx]           # (B, stack, 84, 84) uint8
		next_state = self.frames[rows, nxt_idx]
		action = self.actions[s, i]
		reward = self.rewards[s, i]
		terminated = self.terminated[s, i]

		to = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
		return (
			to(state, torch.uint8),
			to(action, torch.int64),
			to(reward, torch.float32),
			to(next_state, torch.uint8),
			to(terminated, torch.float32),
		)

	# ------------------------------------------------------------------ misc

	def memory_bytes(self):
		return int(sum(a.nbytes for a in (
			self.frames, self.actions, self.rewards,
			self.terminated, self.first, self.is_pad)))
