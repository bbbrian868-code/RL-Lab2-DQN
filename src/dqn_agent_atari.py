"""AtariDQNAgent -- the TA sample `dqn_agent_atari.py` with every TODO filled in.

Search for "[TODO" to find them; each one is closed by "[TODO n] END":

  [TODO 1]  __init__                  initialize env
  [TODO 2]  __init__                  initialize test_env
  [TODO 3]  decide_agent_actions      epsilon-greedy action selection
  [TODO 4]  update_behavior_network   Q(s,a) -> target -> loss -> update
            (step 4's `criterion` is created once, in __init__)

Search for "[EXTRA" for what goes beyond the TODOs: DDQN, Dueling DQN and
parallelized rollout (report sections 3-5).
"""

import random

import numpy as np
import torch
import torch.nn as nn

from atari_wrappers import make_atari_env, make_vector_env
from base_agent import DQNBaseAgent
from models.atari_model import build_network


class AtariDQNAgent(DQNBaseAgent):
	"""DQN and its variants.  `config["variant"]` selects:

	  dqn           y = r + g * max_a' Q_target(s', a')
	  ddqn          y = r + g * Q_target(s', argmax_a' Q_online(s', a'))
	  dueling       dqn target, dueling network
	  dueling_ddqn  ddqn target, dueling network

	Everything else (preprocessing, buffer, schedule, logging) is shared, so
	the report's variant comparisons differ in exactly one thing each.
	"""

	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)

		# ===== [TODO 1] initialize env ===================================
		# Training env(s).  The parallel-rollout variant is the same agent with
		# a vector env behind it.
		if self.num_envs > 1:
			# [EXTRA: parallelized rollout] N envs stepped together
			self.env = make_vector_env(self.env_id, self.num_envs,
			                           asynchronous=config.get("async_envs", True))
			num_actions = self.env.single_action_space.n
		else:
			self.env = make_atari_env(self.env_id, seed=self.seed)
			num_actions = self.env.action_space.n
		# ===== [TODO 1] END ==============================================

		# ===== [TODO 2] initialize test_env ==============================
		# Evaluation env: a separate instance with its own seed sequence, so
		# evaluating never perturbs the training env's RNG or episode state.
		self.test_env = make_atari_env(self.env_id, seed=self.seed + 12345)
		# ===== [TODO 2] END ==============================================

		self.num_actions = int(num_actions)
		# [EXTRA: DDQN] selects the target formula in update_behavior_network
		self.double = self.variant in ("ddqn", "dueling_ddqn")

		# initialize behavior network and target network (TA sample code;
		# build_network returns the TA's AtariNetDQN, or the dueling net for
		# the [EXTRA: Dueling DQN] variants)
		self.behavior_net = build_network(self.variant, self.num_actions).to(self.device)
		self.target_net = build_network(self.variant, self.num_actions).to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		self.target_net.eval()

		# initialize optimizer (TA sample code)
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		# [TODO 4, step 4] `criterion = ???` -- built once here instead of
		# on every update
		self.criterion = nn.SmoothL1Loss() if config.get("loss", "mse") == "huber" else nn.MSELoss()

	# ---------------------------------------------------------------- acting

	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		"""Epsilon-greedy.  Handles a single (4,84,84) obs and a batched
		(N,4,84,84) obs from the vector env."""
		# ===== [TODO 3] get action from behavior net, with epsilon-greedy selection
		batched = np.asarray(observation).ndim == 4

		if not batched:
			if random.random() < epsilon:
				# random action
				return int(action_space.sample())
			with torch.no_grad():
				# greedy action: argmax_a Q(s, a) from the behavior net
				state = torch.as_tensor(np.asarray(observation), dtype=torch.uint8,
				                        device=self.device).unsqueeze(0)
				return int(self.behavior_net(state).argmax(dim=1).item())

		# [EXTRA: parallelized rollout] the same epsilon-greedy rule, applied
		# independently to each of the N observations
		n = len(observation)
		if epsilon >= 1.0:   # warmup: skip the forward pass entirely
			return np.array([action_space.sample() for _ in range(n)], dtype=np.int64)
		with torch.no_grad():
			state = torch.as_tensor(np.asarray(observation), dtype=torch.uint8,
			                        device=self.device)
			greedy = self.behavior_net(state).argmax(dim=1).cpu().numpy()
		explore = np.random.random(n) < epsilon
		if explore.any():
			greedy[explore] = [action_space.sample() for _ in range(int(explore.sum()))]
		return greedy.astype(np.int64)
		# ===== [TODO 3] END ==============================================

	# ---------------------------------------------------------------- update

	def update_behavior_network(self):
		# sample a minibatch of transitions (TA sample code)
		state, action, reward, next_state, done = self.replay_buffer.sample(
			self.batch_size, self.device)

		# ===== [TODO 4] calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		q_value = self.behavior_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

		with torch.no_grad():
			# 2. get max_a Q(s',a) from target net
			if self.double:
				# [EXTRA: DDQN] the online net *picks* the action, the target
				# net *scores* it.  Decoupling selection from evaluation removes
				# the max-operator bias that makes plain DQN over-estimate.
				next_action = self.behavior_net(next_state).argmax(dim=1, keepdim=True)
				q_next = self.target_net(next_state).gather(1, next_action).squeeze(1)
			else:
				q_next = self.target_net(next_state).max(dim=1).values

			# 3. Q_target = r + gamma * max_a Q(s',a); if the episode terminates
			# at next_state, then q_target = reward.
			# `done` is the *terminated* flag only: a truncated episode still
			# bootstraps, because its last state was not absorbing.
			q_target = reward + self.gamma * q_next * (1.0 - done)

		# 4. loss between Q(s,a) and Q_target (criterion is built in __init__)
		loss = self.criterion(q_value, q_target)

		# 5. update behavior net
		self.optim.zero_grad(set_to_none=True)
		loss.backward()
		if self.max_grad_norm:
			nn.utils.clip_grad_norm_(self.behavior_net.parameters(), self.max_grad_norm)
		self.optim.step()

		# TA's `writer.add_scalar('DQN/Loss', ...)`, averaged over log_interval
		# updates and logged as Train/Loss
		self._accumulate_train_scalars(loss.item(), q_value.mean().item())
		# ===== [TODO 4] END ==============================================
