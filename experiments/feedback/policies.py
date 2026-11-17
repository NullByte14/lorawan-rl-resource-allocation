"""Standard tabular Q-learning, copied unchanged from the original pilot."""
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


