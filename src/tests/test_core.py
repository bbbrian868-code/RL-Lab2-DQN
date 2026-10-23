"""Unit tests for the pieces that are easy to get silently wrong.

    python -m pytest tests -q        (run from src/)
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.atari_model import AtariNetDQN, AtariNetDuelingDQN, build_network
from replay_buffer.replay_buffer import ReplayMemory


def frame(v):
	return np.full((84, 84), v, dtype=np.uint8)


def fill_episode(buf, start_value, length, terminated=True, stream=0):
	"""Append one episode of `length` transitions with frames start..start+length."""
	for t in range(length):
		buf.append(
			frame=frame(start_value + t), action=t % 4, reward=1.0,
			terminated=terminated and (t == length - 1),
			episode_done=(t == length - 1),
			episode_start=(t == 0),
			next_frame=frame(start_value + t + 1) if t == length - 1 else None,
			stream=stream,
		)


# ------------------------------------------------------------------- buffer

def test_buffer_shape_and_dtype():
	buf = ReplayMemory(1000)
	fill_episode(buf, 10, 50)
	state, action, reward, next_state, done = buf.sample(8, torch.device("cpu"))
	assert state.shape == (8, 4, 84, 84) and state.dtype == torch.uint8
	assert next_state.shape == (8, 4, 84, 84) and next_state.dtype == torch.uint8
	assert action.shape == (8,) and action.dtype == torch.int64
	assert reward.shape == (8,) and reward.dtype == torch.float32
	assert done.shape == (8,) and done.dtype == torch.float32


def test_buffer_stores_uint8_only():
	buf = ReplayMemory(1000)
	assert buf.frames.dtype == np.uint8
	# 100k x 84 x 84 single frames must stay far under the 6 GB budget.
	big = ReplayMemory(100_000)
	assert big.memory_bytes() < 1.0e9


def test_stack_reconstruction_mid_episode():
	"""state = frames[t-3..t], next_state = frames[t-2..t+1]."""
	buf = ReplayMemory(1000)
	fill_episode(buf, 0, 40)
	idx = np.array([20])
	cur = buf._stack_indices(0, idx)
	nxt = buf._stack_indices(0, idx + 1)
	assert cur.tolist() == [[17, 18, 19, 20]]
	assert nxt.tolist() == [[18, 19, 20, 21]]
	# frame value == index, by construction
	assert buf.frames[0, cur[0], 0, 0].tolist() == [17, 18, 19, 20]


def test_stack_pads_by_repeating_the_reset_frame():
	"""Matches FrameStackObservation(padding_type='reset'): repeat, not zeros."""
	buf = ReplayMemory(1000)
	fill_episode(buf, 100, 10)
	# index 0 is the episode start -> all four slots are that same frame
	assert buf._stack_indices(0, np.array([0])).tolist() == [[0, 0, 0, 0]]
	assert buf._stack_indices(0, np.array([1])).tolist() == [[0, 0, 0, 1]]
	assert buf._stack_indices(0, np.array([2])).tolist() == [[0, 0, 1, 2]]
	assert buf._stack_indices(0, np.array([3])).tolist() == [[0, 1, 2, 3]]


def test_stack_never_crosses_an_episode_boundary():
	buf = ReplayMemory(1000)
	fill_episode(buf, 0, 6)      # slots 0..5 transitions, slot 6 = terminal pad
	fill_episode(buf, 200, 6)    # slots 7..12, slot 13 = pad
	# slot 7 is the second episode's first frame
	assert buf.first[0, 7]
	assert buf._stack_indices(0, np.array([8])).tolist() == [[7, 7, 7, 8]]
	# ... and the pad of episode 1 (slot 6) still stacks inside episode 1
	assert buf._stack_indices(0, np.array([6])).tolist() == [[3, 4, 5, 6]]


def test_pad_entries_are_never_sampled():
	buf = ReplayMemory(1000)
	for k in range(20):
		fill_episode(buf, (k * 11) % 200, 6)
	for _ in range(50):
		s, i = buf._sample_indices(32)
		assert not buf.is_pad[s, i].any()


def test_terminal_transition_has_the_real_next_frame():
	"""The last transition of an episode must see the terminal observation."""
	buf = ReplayMemory(1000)
	fill_episode(buf, 0, 5)                       # transitions 0..4, pad at 5
	nxt = buf._stack_indices(0, np.array([5]))    # next_state of transition 4
	assert buf.frames[0, nxt[0], 0, 0].tolist() == [2, 3, 4, 5]
	assert buf.terminated[0, 4]                   # and it is flagged terminal


def test_truncation_still_bootstraps():
	"""A time-limit end stores terminated=False, so (1-done) keeps the bootstrap."""
	buf = ReplayMemory(1000)
	fill_episode(buf, 0, 5, terminated=False)     # truncated, not terminated
	assert not buf.terminated[0, 4]
	assert buf.is_pad[0, 5]                       # next frame is still available


def test_streams_are_independent():
	buf = ReplayMemory(1000, num_streams=4)
	for s in range(4):
		fill_episode(buf, s * 50, 10, stream=s)
	# stream 2's stack must only contain stream 2's frames (50*2 .. )
	cur = buf._stack_indices(2, np.array([5]))
	vals = buf.frames[2, cur[0], 0, 0].tolist()
	assert vals == [102, 103, 104, 105]
	assert len(buf) == 4 * 11


def test_wraparound_does_not_serve_overwritten_frames():
	buf = ReplayMemory(64)
	for k in range(40):
		fill_episode(buf, (k * 7) % 200, 5)
	assert buf.full[0]
	for _ in range(100):
		s, i = buf._sample_indices(16)
		# every sampled index must be at least stack-1 slots ahead of the
		# write cursor, i.e. its walk-back stays inside live data
		d = (i - buf.pos[0]) % buf.cap
		assert (d >= buf.stack - 1).all()


# ----------------------------------------------------------------- TD target

def td_target(reward, q_next, done, gamma=0.99):
	return reward + gamma * q_next * (1.0 - done)


def test_td_target_does_not_bootstrap_when_done():
	r = torch.tensor([1.0, 1.0])
	q_next = torch.tensor([100.0, 100.0])
	done = torch.tensor([1.0, 0.0])
	y = td_target(r, q_next, done)
	assert torch.allclose(y, torch.tensor([1.0, 1.0 + 0.99 * 100.0]))


def test_dqn_vs_ddqn_targets_differ_as_specified():
	"""DQN takes the target net's max; DDQN scores the online net's argmax."""
	torch.manual_seed(0)
	online = AtariNetDQN(4)
	target = AtariNetDQN(4)
	next_state = torch.randint(0, 255, (16, 4, 84, 84), dtype=torch.uint8)
	with torch.no_grad():
		q_t = target(next_state)
		q_o = online(next_state)
		dqn_q = q_t.max(dim=1).values
		a = q_o.argmax(dim=1, keepdim=True)
		ddqn_q = q_t.gather(1, a).squeeze(1)
	# DDQN's estimate can never exceed DQN's: that is the whole point.
	assert (ddqn_q <= dqn_q + 1e-6).all()
	assert not torch.allclose(ddqn_q, dqn_q)   # the two nets disagree somewhere


# ------------------------------------------------------------------ networks

def test_network_output_shapes():
	x = torch.randint(0, 255, (3, 4, 84, 84), dtype=torch.uint8)
	assert AtariNetDQN(9)(x).shape == (3, 9)
	assert AtariNetDuelingDQN(9)(x).shape == (3, 9)


def test_dueling_advantage_is_mean_centred():
	torch.manual_seed(0)
	net = AtariNetDuelingDQN(6)
	x = torch.randint(0, 255, (5, 4, 84, 84), dtype=torch.uint8)
	with torch.no_grad():
		h = torch.flatten(net.cnn(x.float() / 255.), start_dim=1)
		v, a = net.value(h), net.advantage(h)
		q = net(x)
	# Q - V must have zero mean across actions, i.e. V carries the state value.
	assert torch.allclose((q - v).mean(dim=1), torch.zeros(5), atol=1e-5)
	assert torch.allclose(q, v + a - a.mean(dim=1, keepdim=True), atol=1e-5)


def test_build_network_variant_mapping():
	assert isinstance(build_network("dqn", 4), AtariNetDQN)
	assert not isinstance(build_network("dqn", 4), AtariNetDuelingDQN)
	assert isinstance(build_network("ddqn", 4), AtariNetDQN)
	assert isinstance(build_network("dueling", 4), AtariNetDuelingDQN)
	assert isinstance(build_network("dueling_ddqn", 4), AtariNetDuelingDQN)


def test_target_net_starts_identical():
	torch.manual_seed(0)
	a, b = AtariNetDQN(5), AtariNetDQN(5)
	b.load_state_dict(a.state_dict())
	for pa, pb in zip(a.parameters(), b.parameters()):
		assert torch.equal(pa, pb)


# ------------------------------------------------------- schedules (no env)

class _Sched:
	"""The epsilon / update / target-sync logic lifted out of DQNBaseAgent."""

	def __init__(self, warmup=20000, eps_decay=1_000_000, eps_min=0.1,
	             update_freq=4, update_target_freq=10_000):
		self.warmup_steps = warmup
		self.eps_decay = eps_decay
		self.eps_min = eps_min
		self.update_freq = update_freq
		self.update_target_freq = update_target_freq
		self._update_start_step = warmup
		self._updates_done = 0
		self._last_target_sync = 0
		self.total_time_step = 0
		self.syncs = []

	_epsilon_at = None   # bound below

	def update(self):
		owed = max(0, self.total_time_step - self._update_start_step) // self.update_freq
		n = owed - self._updates_done
		self._updates_done = max(self._updates_done, owed)
		epoch = self.total_time_step // self.update_target_freq
		if epoch > self._last_target_sync:
			self._last_target_sync = epoch
			self.syncs.append(self.total_time_step)
		return n


def _eps(self, step):
	if step < self.warmup_steps:
		return 1.0
	frac = (step - self.warmup_steps) / max(1, self.eps_decay)
	return max(self.eps_min, 1.0 - frac * (1.0 - self.eps_min))


_Sched._epsilon_at = _eps


def test_epsilon_schedule_endpoints():
	s = _Sched()
	assert s._epsilon_at(0) == 1.0
	assert s._epsilon_at(19_999) == 1.0
	assert s._epsilon_at(20_000) == pytest.approx(1.0)
	assert s._epsilon_at(20_000 + 500_000) == pytest.approx(0.55)
	assert s._epsilon_at(20_000 + 1_000_000) == pytest.approx(0.1)
	assert s._epsilon_at(50_000_000) == 0.1   # clamped, never below eps_min


def test_target_sync_happens_on_the_env_step_grid():
	s = _Sched()
	for step in range(1, 45_001):
		s.total_time_step = step
		s.update()
	assert s.syncs == [10_000, 20_000, 30_000, 40_000]


def test_update_ratio_is_preserved_under_parallel_rollout():
	"""8 envs must still average one gradient step per 4 env steps."""
	single = _Sched()
	total = 0
	for step in range(1, 100_001):
		single.total_time_step = step
		total += single.update()

	par = _Sched()
	total_par = 0
	for vec_step in range(1, 100_001 // 8 + 1):
		par.total_time_step = vec_step * 8
		total_par += par.update()

	assert total == (100_000 - 20_000) // 4
	assert abs(total - total_par) <= 2
	# and the target net syncs at the same env steps in both
	assert single.syncs == par.syncs


def test_parallel_rollout_target_sync_not_skipped_by_large_jumps():
	"""num_envs > update_target_freq spacing must not silently drop a sync."""
	s = _Sched(update_target_freq=100)
	for vec_step in range(1, 51):
		s.total_time_step = vec_step * 64      # jumps straddle several epochs
		s.update()
	assert s.syncs, "no target sync happened at all"
	# monotonically increasing, one per crossed epoch boundary
	assert s.syncs == sorted(s.syncs)
	assert len(set(s.syncs)) == len(s.syncs)
