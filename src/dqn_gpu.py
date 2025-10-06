"""
Deep Q-Network (DQN) implementation with PyTorch for GPU acceleration.
This is an alternative to tabular Q-learning that can benefit from GPU processing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import os

from bdi import Actions, Beliefs

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DQN using device: {DEVICE}")


class DQN(nn.Module):
    """Deep Q-Network architecture for soccer agents."""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 128, 64]):
        """
        Initialize DQN.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DQN, self).__init__()
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Regularization
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', 
                                   field_names=['state', 'action', 'reward', 'next_state', 'done'])
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample(self, batch_size: int):
        """Sample random batch of experiences."""
        experiences = random.sample(self.buffer, k=batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences if e is not None]).to(DEVICE)
        actions = torch.LongTensor([e.action for e in experiences if e is not None]).to(DEVICE)
        rewards = torch.FloatTensor([e.reward for e in experiences if e is not None]).to(DEVICE)
        next_states = torch.FloatTensor([e.next_state for e in experiences if e is not None]).to(DEVICE)
        dones = torch.BoolTensor([e.done for e in experiences if e is not None]).to(DEVICE)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN agent with GPU acceleration for soccer environment."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        update_every: int = 4,
        buffer_size: int = 100000,
        tau: float = 0.001  # Soft update parameter
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_size: Dimension of state representation
            action_size: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for exploration
            epsilon_min: Minimum exploration rate
            batch_size: Size of training batches
            update_every: Frequency of learning updates
            buffer_size: Size of replay buffer
            tau: Soft update rate for target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_every = update_every
        self.tau = tau
        
        # Q-Networks
        self.q_network_local = DQN(state_size, action_size).to(DEVICE)
        self.q_network_target = DQN(state_size, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Step counter for learning frequency
        self.t_step = 0
        
        # Performance tracking
        self.loss_history = []
        self.q_value_history = []
    
    def beliefs_to_state(self, beliefs: Beliefs) -> np.ndarray:
        """
        Convert beliefs to normalized state vector for neural network.
        
        Args:
            beliefs: Agent's beliefs
            
        Returns:
            Normalized state vector
        """
        # Extract and normalize belief values
        state = []
        
        # Distances (normalized by reasonable field distances)
        state.append((beliefs.distance_to_ball or 0) / 100.0)
        state.append((beliefs.distance_to_goal or 0) / 100.0)
        state.append((beliefs.distance_to_home_goal or 0) / 100.0)
        state.append((beliefs.distance_to_opponent or 0) / 100.0)
        
        # Boolean states (0 or 1)
        state.append(1.0 if beliefs.teammate_open else 0.0)
        state.append(1.0 if beliefs.goal_open else 0.0)
        state.append(1.0 if beliefs.has_ball_possession else 0.0)
        state.append(1.0 if beliefs.team_has_possession else 0.0)
        state.append(1.0 if beliefs.in_attacking_third else 0.0)
        state.append(1.0 if beliefs.in_defensive_third else 0.0)
        state.append(1.0 if beliefs.opponent_threatening else 0.0)
        
        # Teammate spacing (new dispersion features)
        state.append(min(beliefs.teammates_too_close, 5) / 5.0)  # Normalize by max expected
        state.append(len(beliefs.teammates_nearby) / 10.0)  # Normalize by max teammates
        state.append((beliefs.closest_teammate_distance or 100) / 100.0)
        
        return np.array(state, dtype=np.float32)
    
    def action_to_index(self, action: Actions) -> int:
        """Convert Actions enum to integer index."""
        return list(Actions).index(action)
    
    def index_to_action(self, index: int) -> Actions:
        """Convert integer index to Actions enum."""
        return list(Actions)[index]
    
    def select_action(self, beliefs: Beliefs, training: bool = True) -> Actions:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            beliefs: Current beliefs
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        state = self.beliefs_to_state(beliefs)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            action_index = random.randint(0, self.action_size - 1)
        else:
            self.q_network_local.eval()
            with torch.no_grad():
                q_values = self.q_network_local(state_tensor)
                action_index = q_values.argmax().item()
            self.q_network_local.train()
        
        return self.index_to_action(action_index)
    
    def step(self, beliefs: Beliefs, action: Actions, reward: float, 
             next_beliefs: Beliefs, done: bool):
        """
        Save experience and learn from batch of experiences.
        
        Args:
            beliefs: Current beliefs (state)
            action: Action taken
            reward: Reward received
            next_beliefs: Next beliefs (next state)
            done: Whether episode ended
        """
        # Convert to state vectors
        state = self.beliefs_to_state(beliefs)
        next_state = self.beliefs_to_state(next_beliefs)
        action_index = self.action_to_index(action)
        
        # Save experience in replay buffer
        self.memory.add(state, action_index, reward, next_state, done)
        
        # Learn every update_every steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)
    
    def learn(self, experiences: Tuple):
        """
        Learn from batch of experiences using DQN algorithm.
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get Q values for current states
        q_expected = self.q_network_local(states).gather(1, actions.unsqueeze(1))
        
        # Get max Q values for next states from target network
        q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Calculate target Q values
        q_targets = rewards.unsqueeze(1) + (self.gamma * q_targets_next * ~dones.unsqueeze(1))
        
        # Calculate loss
        loss = F.mse_loss(q_expected, q_targets)
        
        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.q_network_local, self.q_network_target, self.tau)
        
        # Track performance
        self.loss_history.append(loss.item())
        self.q_value_history.append(q_expected.mean().item())
    
    def soft_update(self, local_model, target_model, tau):
        """
        Soft update target network: θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: Weights will be copied from this model
            target_model: Weights will be copied to this model
            tau: Interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """Save trained model."""
        torch.save({
            'local_state_dict': self.q_network_local.state_dict(),
            'target_state_dict': self.q_network_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'q_value_history': self.q_value_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=DEVICE)
            self.q_network_local.load_state_dict(checkpoint['local_state_dict'])
            self.q_network_target.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.loss_history = checkpoint.get('loss_history', [])
            self.q_value_history = checkpoint.get('q_value_history', [])
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'avg_q_value': np.mean(self.q_value_history[-100:]) if self.q_value_history else 0,
            'device': str(DEVICE)
        }


def benchmark_dqn_vs_tabular():
    """Benchmark DQN vs tabular Q-learning performance."""
    import sys
    sys.path.append('.')
    
    from agents import Agent, Team
    from env import Environment
    import time
    
    print(f"=== DQN vs Tabular Q-Learning Benchmark ===")
    print(f"Using device: {DEVICE}")
    
    # Create test environment
    env = Environment()
    
    # Test configurations
    configs = {
        'DQN_GPU': {'use_dqn': True, 'agents': 6},
        'Tabular': {'use_dqn': False, 'agents': 6},
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\nTesting {config_name}...")
        
        # Create agents
        agents = []
        for i in range(config['agents']):
            team = Team.BLUE if i < config['agents'] // 2 else Team.WHITE
            role = 'midfielder'
            pos = np.array([(-15 if i < config['agents'] // 2 else 15), (i % 2 - 0.5) * 10])
            
            if config['use_dqn']:
                # Create DQN agent
                state_size = 14  # Size of state vector from beliefs_to_state
                action_size = len(Actions)
                agent = DQNAgent(state_size, action_size)
                agents.append(agent)
            else:
                # Create regular agent
                agent = Agent(env, team, role=role, pos=pos)
                agents.append(agent)
        
        # Benchmark episode performance
        episodes = 5
        total_time = 0
        total_steps = 0
        
        for episode in range(episodes):
            env.reset()
            
            if not config['use_dqn']:
                for agent in agents:
                    agent.reset_episode()
            
            episode_start = time.time()
            
            # Run episode
            for step in range(100):  # Max steps per episode
                if config['use_dqn']:
                    # DQN agents
                    actions = []
                    for agent in agents:
                        # Create dummy beliefs for benchmarking
                        beliefs = type('Beliefs', (), {
                            'distance_to_ball': np.random.uniform(0, 50),
                            'distance_to_goal': np.random.uniform(0, 100),
                            'distance_to_home_goal': np.random.uniform(0, 100),
                            'distance_to_opponent': np.random.uniform(0, 50),
                            'teammate_open': np.random.choice([True, False]),
                            'goal_open': np.random.choice([True, False]),
                            'has_ball_possession': np.random.choice([True, False]),
                            'team_has_possession': np.random.choice([True, False]),
                            'in_attacking_third': np.random.choice([True, False]),
                            'in_defensive_third': np.random.choice([True, False]),
                            'opponent_threatening': np.random.choice([True, False]),
                            'teammates_too_close': np.random.randint(0, 3),
                            'teammates_nearby': [],
                            'closest_teammate_distance': np.random.uniform(5, 20)
                        })()
                        
                        action = agent.select_action(beliefs, training=True)
                        actions.append(action)
                else:
                    # Regular agents
                    actions = [agent.act() for agent in agents]
                
                # Step environment (simplified for benchmarking)
                obs, rewards, done, info = env.step(actions)
                total_steps += 1
                
                if done:
                    break
            
            total_time += time.time() - episode_start
        
        # Calculate results
        avg_time_per_episode = total_time / episodes
        avg_steps_per_second = total_steps / total_time if total_time > 0 else 0
        
        results[config_name] = {
            'avg_episode_time': avg_time_per_episode,
            'steps_per_second': avg_steps_per_second,
            'total_time': total_time,
            'episodes': episodes,
            'agents': config['agents']
        }
        
        print(f"  Avg episode time: {avg_time_per_episode:.4f}s")
        print(f"  Steps per second: {avg_steps_per_second:.1f}")
    
    # Print comparison
    print(f"\n=== Comparison Results ===")
    tabular_time = results['Tabular']['avg_episode_time']
    dqn_time = results['DQN_GPU']['avg_episode_time']
    
    if dqn_time > 0:
        speedup = tabular_time / dqn_time
        print(f"DQN vs Tabular speedup: {speedup:.2f}x")
        print(f"GPU utilization: {torch.cuda.utilization() if torch.cuda.is_available() else 'N/A'}%")
    
    return results


if __name__ == "__main__":
    benchmark_dqn_vs_tabular()