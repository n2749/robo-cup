import random
import math
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Any
from bdi import Actions


class QLearningPolicy:
    """
    Q-learning policy adapted for soccer simulation with BDI architecture.
    Handles discrete state-action space for soccer agent decision making.
    """
    
    def __init__(
        self,
        actions: List[Actions],
        alpha: float = 0.10,
        gamma: float = 0.95,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995
    ):
        """
        Initialize Q-learning policy for soccer agents.
        
        Args:
            actions: List of available actions (from Actions enum)
            alpha: Learning rate
            gamma: Discount factor
            eps_start: Initial exploration rate
            eps_end: Minimum exploration rate
            eps_decay: Exploration decay rate
        """
        self.actions = actions
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.episode_count = 0
        
    def _state_key(self, beliefs) -> Tuple:
        """
        Convert beliefs to a discrete state representation for Q-table indexing.
        
        Args:
            beliefs: Agent's current beliefs
            
        Returns:
            Tuple representing the state
        """
        # Discretize continuous values for Q-table
        MAX_BALL_DISTANCE = math.sqrt(100**2 + 65**2)
        MAX_GOAL_DISTANCE = math.sqrt(100**2 + 65**2)

        def log_normalize(d, v_max):
            epsilon = 1e-9
            d = d if d is not None else 0
            log = math.log(max(d / v_max, epsilon), 1/2)
            norm = min(math.floor(log), 10)
            return norm

        distance_to_ball_bucket = log_normalize(beliefs.distance_to_ball, MAX_BALL_DISTANCE)
        distance_to_goal_bucket = log_normalize(beliefs.distance_to_goal, MAX_GOAL_DISTANCE)
        distance_to_home_bucket = log_normalize(beliefs.distance_to_home_goal, MAX_GOAL_DISTANCE)
        distance_to_opponent_bucket = log_normalize(beliefs.distance_to_opponent, MAX_BALL_DISTANCE)

        # print(f"distance_to_ball={beliefs.distance_to_ball}, v_max={MAX_BALL_DISTANCE}, distance_to_ball_bucket={distance_to_ball_bucket}")

        # Boolean states
        teammate_open = bool(beliefs.teammate_open) if beliefs.teammate_open is not None else False
        goal_open = bool(beliefs.goal_open) if beliefs.goal_open is not None else False
        
        return (
            distance_to_ball_bucket,
            distance_to_goal_bucket, 
            distance_to_home_bucket,
            distance_to_opponent_bucket,
            teammate_open,
            goal_open
        )
    
    def _q_key(self, state: Tuple, action: Actions) -> Tuple:
        """Create key for Q-table lookup"""
        return (state, action)
    
    def select_action(self, beliefs, greedy: bool = False) -> Actions:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            beliefs: Agent's current beliefs
            greedy: If True, always select best action (no exploration)
            
        Returns:
            Selected action
        """
        state = self._state_key(beliefs)
        
        if not greedy and random.random() < self.eps:
            # Explore: random action
            return random.choice(self.actions)
        else:
            # Exploit: best action
            best_action = None
            best_q_value = float('-inf')
            
            for action in self.actions:
                q_value = self.Q[self._q_key(state, action)]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            return best_action if best_action is not None else random.choice(self.actions)
    
    def update(self, prev_beliefs, action: Actions, reward: float, curr_beliefs, done: bool):
        """
        Update Q-values using Q-learning update rule.
        
        Args:
            prev_beliefs: Previous beliefs (state)
            action: Action taken
            reward: Reward received
            curr_beliefs: Current beliefs (next state)
            done: Whether episode is finished
        """
        prev_state = self._state_key(prev_beliefs)
        curr_state = self._state_key(curr_beliefs)
        
        q_key = self._q_key(prev_state, action)
        
        if done:
            # Terminal state
            td_target = reward
        else:
            # Find max Q-value for next state
            max_next_q = max(
                self.Q[self._q_key(curr_state, next_action)] 
                for next_action in self.actions
            )
            td_target = reward + self.gamma * max_next_q
        
        # Q-learning update
        td_error = td_target - self.Q[q_key]
        self.Q[q_key] += self.alpha * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        
    def end_episode(self):
        """Call at end of each episode"""
        self.episode_count += 1
        self.decay_epsilon()
        
    def get_stats(self) -> dict:
        """Get learning statistics"""
        return {
            'epsilon': self.eps,
            'q_table_size': len(self.Q),
            'episode_count': self.episode_count
        }


class MultiAgentQLearning:
    """
    Manages Q-learning for multiple agents in the soccer simulation.
    """
    
    def __init__(self, num_agents: int, actions: List[Actions], **kwargs):
        """
        Initialize multi-agent Q-learning.
        
        Args:
            num_agents: Number of agents to manage
            actions: Available actions for all agents
            **kwargs: Parameters passed to QLearningPolicy
        """
        self.policies = [QLearningPolicy(actions, **kwargs) for _ in range(num_agents)]
        
    def select_actions(self, beliefs_list: List, greedy: bool = False) -> List[Actions]:
        """Select actions for all agents"""
        return [
            policy.select_action(beliefs, greedy)
            for policy, beliefs in zip(self.policies, beliefs_list)
        ]
    
    def update_all(self, prev_beliefs_list: List, actions: List[Actions], 
                   rewards: List[float], curr_beliefs_list: List, done: bool):
        """Update all agent policies"""
        for i, policy in enumerate(self.policies):
            policy.update(
                prev_beliefs_list[i], actions[i], rewards[i], 
                curr_beliefs_list[i], done
            )
    
    def end_episode(self):
        """End episode for all policies"""
        for policy in self.policies:
            policy.end_episode()
            
    def get_combined_stats(self) -> dict:
        """Get combined statistics for all agents"""
        stats = {}
        for i, policy in enumerate(self.policies):
            agent_stats = policy.get_stats()
            for key, value in agent_stats.items():
                if key not in stats:
                    stats[key] = []
                stats[key].append(value)
        
        # Calculate averages
        for key in stats:
            if isinstance(stats[key][0], (int, float)):
                stats[f'avg_{key}'] = np.mean(stats[key])
        
        return stats
