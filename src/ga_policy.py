#!/usr/bin/env python3
"""
Genetic Algorithm policy adapter for BDI soccer agents.

This policy is a drop-in replacement for the tabular QLearningPolicy used by
agents. It matches the minimal interface expected by Agent.plan()/Agent.learn():
- actions: list of available Actions (may be temporarily set by Agent.plan)
- select_action(beliefs, greedy=False) -> Actions
- update(...): no-op for GA during episode
- end_episode(): handles epsilon decay
- attributes: Q (dict), alpha, gamma, eps, eps_end, eps_decay, episode_count

Chromosome encodes a linear policy: for each action, a weight vector over a
fixed feature vector derived from beliefs. Action score = w · x; the best action
is selected (with epsilon exploration).
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import defaultdict

from bdi import Actions


def _discretize_beliefs(beliefs) -> Tuple[int, int, int, int, bool, bool]:
    """Mirror qlearning._state_key discretization for consistency."""
    d_ball = min(4, int(beliefs.distance_to_ball / 10.0)) if beliefs.distance_to_ball is not None else 0
    d_goal = min(4, int(beliefs.distance_to_goal / 20.0)) if beliefs.distance_to_goal is not None else 0
    d_home = min(4, int(beliefs.distance_to_home_goal / 20.0)) if beliefs.distance_to_home_goal is not None else 0
    d_opp = min(4, int(beliefs.distance_to_opponent / 10.0)) if beliefs.distance_to_opponent is not None else 0
    teammate_open = bool(beliefs.teammate_open) if beliefs.teammate_open is not None else False
    goal_open = bool(beliefs.goal_open) if beliefs.goal_open is not None else False
    return (d_ball, d_goal, d_home, d_opp, teammate_open, goal_open)


def _feature_vector(beliefs) -> np.ndarray:
    """Build a feature vector x from beliefs for linear policy scoring.
    Features: [bias, d_ball, d_goal, d_home, d_opp, teammate_open, goal_open]
    """
    d_ball, d_goal, d_home, d_opp, teammate_open, goal_open = _discretize_beliefs(beliefs)
    return np.array([
        1.0,                       # bias
        float(d_ball),
        float(d_goal),
        float(d_home),
        float(d_opp),
        1.0 if teammate_open else 0.0,
        1.0 if goal_open else 0.0,
    ], dtype=np.float32)


class GeneticPolicy:
    """Chromosome-based linear policy over belief-derived features."""

    def __init__(
        self,
        actions: List[Actions],
        num_features: int = 7,
        weights: Optional[np.ndarray] = None,
        eps_start: float = 0.05,
        eps_end: float = 0.0,
        eps_decay: float = 0.995,
    ):
        self.actions: List[Actions] = list(actions)
        self._all_actions: List[Actions] = list(Actions)  # fixed order for chromosome layout
        self.num_features = num_features
        self.num_actions = len(self._all_actions)

        # Weights shape: [num_actions, num_features]
        if weights is None:
            self.weights = np.random.normal(0.0, 0.5, size=(self.num_actions, self.num_features)).astype(np.float32)
        else:
            w = np.asarray(weights, dtype=np.float32)
            assert w.shape == (self.num_actions, self.num_features), (
                f"weights must be shape ({self.num_actions}, {self.num_features})"
            )
            self.weights = w.copy()

        # API compatibility attributes (placeholders)
        # Keep a defaultdict(float) so code that indexes Q[_q_key(...)] works safely.
        self.Q: Dict = defaultdict(float)
        self.alpha = 0.0
        self.gamma = 0.0
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.episode_count = 0

    # --- Policy API ---
    def select_action(self, beliefs, greedy: bool = False) -> Actions:
        # Epsilon exploration over current allowed actions
        if not greedy and np.random.random() < self.eps and len(self.actions) > 1:
            return np.random.choice(self.actions)

        x = _feature_vector(beliefs)  # [num_features]

        # Score only over currently allowed actions
        best_action = None
        best_score = -1e9
        for a in self.actions:
            idx = self._all_actions.index(a)
            score = float(np.dot(self.weights[idx], x))
            if score > best_score:
                best_score = score
                best_action = a
        return best_action if best_action is not None else np.random.choice(self.actions)

    def update(self, prev_beliefs, action: Actions, reward: float, curr_beliefs, done: bool):
        # GA has no within-episode update
        return

    # Provide QLearningPolicy-compatible helpers so other code paths can read Q-values safely
    def _state_key(self, beliefs) -> Tuple:
        return _discretize_beliefs(beliefs)

    def _q_key(self, state: Tuple, action: Actions) -> Tuple:
        return (state, action)

    def end_episode(self):
        self.episode_count += 1
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def get_stats(self) -> dict:
        return {
            'epsilon': float(self.eps),
            'chromosome_size': int(self.num_actions * self.num_features),
            'episode_count': int(self.episode_count),
        }

    # --- GA utilities ---
    def to_gene(self) -> np.ndarray:
        return self.weights.flatten().copy()

    @classmethod
    def from_gene(
        cls,
        gene: np.ndarray,
        eps_start: float = 0.0,
        eps_end: float = 0.0,
        eps_decay: float = 1.0,
    ) -> "GeneticPolicy":
        g = np.asarray(gene, dtype=np.float32)
        num_actions = len(Actions)
        num_features = g.size // num_actions
        weights = g.reshape((num_actions, num_features)).copy()
        return cls(actions=list(Actions), num_features=num_features, weights=weights,
                   eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)

    def clone(self) -> "GeneticPolicy":
        newp = GeneticPolicy(actions=list(self.actions), num_features=self.num_features,
                             weights=self.weights.copy(), eps_start=self.eps,
                             eps_end=self.eps_end, eps_decay=self.eps_decay)
        return newp


# Simple GA operators
class GAOps:
    @staticmethod
    def init_population(pop_size: int, num_features: int = 7) -> np.ndarray:
        num_actions = len(Actions)
        shape = (pop_size, num_actions * num_features)
        return np.random.normal(0.0, 0.5, size=shape).astype(np.float32)

    @staticmethod
    def tournament_select(pop: np.ndarray, fitness: np.ndarray, k: int = 3) -> int:
        idxs = np.random.choice(len(pop), size=min(k, len(pop)), replace=False)
        best = idxs[0]
        for i in idxs[1:]:
            if fitness[i] > fitness[best]:
                best = i
        return int(best)

    @staticmethod
    def uniform_crossover(p1: np.ndarray, p2: np.ndarray, p: float = 0.5) -> np.ndarray:
        mask = (np.random.random(size=p1.shape) < p)
        child = np.where(mask, p1, p2)
        return child.astype(np.float32)

    @staticmethod
    def mutate(gene: np.ndarray, sigma: float = 0.1, rate: float = 0.1) -> np.ndarray:
        mask = (np.random.random(size=gene.shape) < rate)
        noise = np.random.normal(0.0, sigma, size=gene.shape).astype(np.float32)
        out = gene.copy()
        out[mask] += noise[mask]
        return out
