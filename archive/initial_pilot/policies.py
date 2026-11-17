"""Tabular Q-learning and a documented MIX-MAB adaptation for the pilot.

MIX-MAB source: Azizi et al., Algorithm 1 and Section III,
https://arxiv.org/pdf/2206.03401 (2022), PDF pages 3-4.

The common simulator supplies the radio action set, observations and delayed
feedback. MIX-MAB ignores state, scalar reward and energy. The main pilot
chooses an action for every uplink, then supplies delayed binary packet
outcomes from valid reports. Channel selection is outside this pilot's action
set, and observations arrive later than in the published method. Label the
method as a matched-option, delayed-feedback MIX-MAB adaptation rather than an
exact reproduction of the original method or experiments. The API optionally
accepts mean success fractions for separate block-feedback experiments; that
unused mode changes decision cadence and needs its own adaptation label.

Implementation choices resolving ambiguities in Algorithm 1:
* Five complete round-robin passes follow the prose and l_EXP=5, rather than
  the pseudocode's <= comparison, which would make six passes.
* Probabilities are recomputed for all currently active arms and normalized
  after elimination. Removed arms remain removed until a round-robin restart.
* Selection counts trigger a restart after a count exceeds alpha*l_EE.
  Restart clears counts and reopens arms, retaining learned weights.
* Delayed updates use the action's stored EXP3 probability from selection time.
  During round robin, it is the hypothetical EXP3 probability, because round
  robin has no stochastic selection probability. The paper updates immediately
  using its calculated PDF. No feedback is invented when a report is absent.
"""

from collections import defaultdict, deque
from math import e, log, sqrt

import numpy as np


class QPolicy:
    """Standard tabular Q-learning; state construction and reward are external."""

    def __init__(self, n_actions=30, alpha=0.1, gamma=0.9, epsilon=0.1,
                 update_enabled=True):
        if n_actions < 1 or not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("Invalid Q-learning parameters")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must lie between zero and one")
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.update_enabled = bool(update_enabled)
        self.q = {}
        self.update_count = 0

    def _row(self, state):
        key = tuple(state)
        row = self.q.get(key)
        if row is None:
            row = np.zeros(self.n_actions, dtype=np.float32)
            if self.update_enabled:
                self.q[key] = row
        return row

    def choose(self, state, rng):
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        row = self._row(state)
        return int(rng.choice(np.flatnonzero(row == row.max())))

    def update(self, state, action, reward, next_state, success=None,
               energy_norm=None):
        if not self.update_enabled:
            return
        if not 0 <= action < self.n_actions or not np.isfinite(reward):
            raise ValueError("Invalid action or non-finite reward")
        row = self._row(state)
        target = float(reward) + self.gamma * float(self._row(next_state).max())
        row[action] += self.alpha * (target - float(row[action]))
        self.update_count += 1

    @property
    def q_entries(self):
        return len(self.q) * self.n_actions

    @property
    def policy_size_bytes(self):
        """Numeric Q payload only; excludes Python container/allocator overhead."""
        return sum(row.nbytes for row in self.q.values())


class MixMAB:
    """MIX-MAB-inspired adaptation with reported reception rewards in [0, 1].

    The main simulator chooses per uplink and updates each packet's stored
    action when its binary outcome is subsequently reported. Counts, horizon
    and restart thresholds therefore measure packet decisions. Optional mean
    success fractions support separate block-feedback studies that choose an
    action once per report cycle. Such a study treats each block as one bandit
    observation despite unequal block sizes and needs a separate label.

    One instance represents one end device. Calls to update must preserve FIFO
    ordering for repeated choices of the same action. The simulator should
    deliver batches in sequence order, including late batches. It must discard
    duplicated feedback before calling update. If an attempt expires without
    feedback, call discard_pending(action), in sequence order, to remove its
    stored selection probability without generating a reward.

    horizon is the predeclared expected number of decision rounds per device,
    used only in the paper's fixed gamma formula. It must not be derived from
    realized future packet counts. Default parameters are l_EXP=5, l_EE=100,
    and initial alpha=1 from the paper. Logs keep weights numerically stable.
    """

    def __init__(self, n_actions=30, horizon=5000, l_exp=5, l_ee=100):
        if n_actions < 1 or horizon < 1 or l_exp < 1 or l_ee < 1:
            raise ValueError("MIX-MAB dimensions and horizons must be positive")
        self.n_actions = int(n_actions)
        self.horizon = int(horizon)
        self.gamma = min(1.0, sqrt(self.n_actions * log(self.n_actions)
                                   / ((e - 1) * self.horizon)))
        self.l_exp = int(l_exp)
        self.l_ee = int(l_ee)
        self.alpha = 1
        self.epoch = 0
        self.counts = np.zeros(self.n_actions, dtype=np.int64)
        self.log_weights = np.zeros(self.n_actions, dtype=np.float64)
        self.active = np.ones(self.n_actions, dtype=bool)
        self.pending = defaultdict(deque)
        self.round_robin = 0
        self.update_count = 0
        self.restart_count = 0

    def probabilities(self):
        """Normalized EXP3 probabilities over arms not currently eliminated."""
        weights = np.exp(self.log_weights - self.log_weights.max())
        probabilities = (1 - self.gamma) * weights / weights.sum()
        probabilities += self.gamma / self.n_actions
        probabilities[~self.active] = 0.0
        total = probabilities.sum()
        if total <= 0 or not np.isfinite(total):
            raise FloatingPointError("MIX-MAB probability normalization failed")
        return probabilities / total

    def choose(self, state, rng):
        probabilities = self.probabilities()
        if self.counts.min() < self.l_exp:
            action = self.round_robin % self.n_actions
            self.round_robin += 1
        else:
            action = int(rng.choice(self.n_actions, p=probabilities))
        self.pending[action].append((float(probabilities[action]), self.epoch))
        self.counts[action] += 1
        if self.counts[action] > self.alpha * self.l_ee:
            self.counts.fill(0)
            self.active.fill(True)
            self.round_robin = 0
            self.alpha += 1
            self.epoch += 1
            self.restart_count += 1
        return action

    def update(self, state, action, reward, next_state, success=None,
               energy_norm=None):
        if success is None or not np.isfinite(success) or not 0 <= success <= 1:
            raise ValueError("MIX-MAB adaptation requires a success value in [0, 1]")
        if not self.pending[action]:
            raise ValueError("No pending choice matches this feedback")
        probability, epoch = self.pending[action].popleft()
        if probability <= 0:
            raise FloatingPointError("Selected action had zero EXP3 probability")
        self.log_weights[action] += (self.gamma * float(success)
                                     / (self.n_actions * probability))
        self.log_weights -= self.log_weights.max()
        np.maximum(self.log_weights, -700.0, out=self.log_weights)
        self.update_count += 1
        probabilities = self.probabilities()
        if (epoch == self.epoch and self.counts[action] > self.l_exp
                and probabilities[action] < 0.5 * probabilities.max()
                and self.active.sum() > 1):
            self.active[action] = False

    def discard_pending(self, action):
        """Forget an expired unknown outcome, without success/failure learning."""
        if not self.pending[action]:
            raise ValueError("No pending choice to discard")
        self.pending[action].popleft()

    @property
    def in_round_robin(self):
        """Whether the next choice is still in the current exploration pass."""
        return bool(self.counts.min() < self.l_exp)

    @property
    def min_selection_count(self):
        """Minimum number of packet choices per arm since the latest restart."""
        return int(self.counts.min())

    @property
    def q_entries(self):
        return 0

    @property
    def policy_size_bytes(self):
        """Numeric policy payload only, excluding pending metadata and Python."""
        return self.counts.nbytes + self.log_weights.nbytes + self.active.nbytes
