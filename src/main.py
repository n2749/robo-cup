
from agents import Defender, Attacker, Team, FieldDistribution
from env import Environment
from bdi import Actions
import numpy as np
import json
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import time


class TrainingConfig:
    """Configuration class for training parameters"""
    def __init__(
        self,
        num_episodes: int = 1000,
        checkpoint_frequency: int = 100,
        stats_frequency: int = 10,
        save_directory: str = "checkpoints",
        max_episode_steps: int = 600,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        early_stopping_patience: int = 200,
        target_win_rate: float = 0.8
    ):
        self.num_episodes = num_episodes
        self.checkpoint_frequency = checkpoint_frequency
        self.stats_frequency = stats_frequency
        self.save_directory = save_directory
        self.max_episode_steps = max_episode_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.early_stopping_patience = early_stopping_patience
        self.target_win_rate = target_win_rate
        
        # Create save directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)


class TrainingStats:
    """Class to track and manage training statistics"""
    def __init__(self):
        self.episode_rewards = []  # List of total rewards per episode
        self.episode_lengths = []  # List of episode lengths
        self.win_rates = []  # Win rate over sliding window
        self.goals_scored = {'blue': [], 'white': []}  # Goals per episode
        self.q_table_sizes = []  # Size of Q-tables over time
        self.exploration_rates = []  # Epsilon values over time
        self.collision_counts = []  # Collisions per episode
        self.episode_times = []  # Real time taken per episode
        self.start_time = time.time()
        self.best_episode_reward = float('-inf')
        self.best_win_rate = 0.0
        
    def update_episode_stats(self, episode_reward: float, episode_length: int, 
                           goals: dict, q_table_size: int, exploration_rate: float,
                           collision_count: int, episode_time: float):
        """Update statistics for completed episode"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.goals_scored['blue'].append(goals.get('blue', 0))
        self.goals_scored['white'].append(goals.get('white', 0))
        self.q_table_sizes.append(q_table_size)
        self.exploration_rates.append(exploration_rate)
        self.collision_counts.append(collision_count)
        self.episode_times.append(episode_time)
        
        # Update best stats
        if episode_reward > self.best_episode_reward:
            self.best_episode_reward = episode_reward
            
    def calculate_win_rate(self, window_size: int = 100) -> float:
        """Calculate win rate over last window_size episodes"""
        if len(self.goals_scored['blue']) < window_size:
            window_size = len(self.goals_scored['blue'])
        
        if window_size == 0:
            return 0.0
            
        recent_blue = self.goals_scored['blue'][-window_size:]
        recent_white = self.goals_scored['white'][-window_size:]
        
        wins = sum(1 for b, w in zip(recent_blue, recent_white) if b > w)
        win_rate = wins / window_size
        
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            
        return win_rate
    
    def get_summary_stats(self, window_size: int = 100) -> dict:
        """Get summary of recent training statistics"""
        if not self.episode_rewards:
            return {}
            
        recent_rewards = self.episode_rewards[-window_size:]
        recent_lengths = self.episode_lengths[-window_size:]
        win_rate = self.calculate_win_rate(window_size)
        
        return {
            'episodes_completed': len(self.episode_rewards),
            'avg_reward': np.mean(recent_rewards),
            'avg_episode_length': np.mean(recent_lengths),
            'win_rate': win_rate,
            'best_win_rate': self.best_win_rate,
            'best_episode_reward': self.best_episode_reward,
            'current_exploration_rate': self.exploration_rates[-1] if self.exploration_rates else 0,
            'avg_q_table_size': np.mean(self.q_table_sizes[-window_size:]) if self.q_table_sizes else 0,
            'total_training_time': time.time() - self.start_time,
            'avg_episode_time': np.mean(self.episode_times[-window_size:]) if self.episode_times else 0
        }


def save_checkpoint(config: FieldDistribution, episode: int, stats: TrainingStats, 
                   training_config: TrainingConfig) -> str:
    """Save training progress to checkpoint file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"checkpoint_ep{episode}_{timestamp}.pkl"
    checkpoint_path = os.path.join(training_config.save_directory, checkpoint_filename)
    
    # Extract Q-tables and agent states
    checkpoint_data = {
        'episode': episode,
        'timestamp': timestamp,
        'training_config': training_config.__dict__,
        'stats': {
            'episode_rewards': stats.episode_rewards,
            'episode_lengths': stats.episode_lengths,
            'win_rates': stats.win_rates,
            'goals_scored': stats.goals_scored,
            'q_table_sizes': stats.q_table_sizes,
            'exploration_rates': stats.exploration_rates,
            'collision_counts': stats.collision_counts,
            'episode_times': stats.episode_times,
            'start_time': stats.start_time,
            'best_episode_reward': stats.best_episode_reward,
            'best_win_rate': stats.best_win_rate
        },
        'agents': []
    }
    
    # Save each agent's state and Q-table
    for i, agent in enumerate(config.agents):
        agent_data = {
            'role': agent.role,
            'team': agent.team.name,
            'position': agent.pos.tolist(),
            'q_table': dict(agent.q_policy.Q),  # Convert defaultdict to regular dict
            'q_policy_params': {
                'alpha': agent.q_policy.alpha,
                'gamma': agent.q_policy.gamma,
                'eps': agent.q_policy.eps,
                'eps_end': agent.q_policy.eps_end,
                'eps_decay': agent.q_policy.eps_decay,
                'episode_count': agent.q_policy.episode_count
            },
            'episode_rewards': agent.episode_rewards,
            'cumulative_reward': agent.cumulative_reward
        }
        checkpoint_data['agents'].append(agent_data)
    
    # Save to file
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Also save a JSON summary for easy inspection
    summary_path = os.path.join(training_config.save_directory, f"summary_ep{episode}_{timestamp}.json")
    summary_data = {
        'episode': episode,
        'timestamp': timestamp,
        'stats_summary': stats.get_summary_stats(),
        'agent_count': len(config.agents),
        'checkpoint_file': checkpoint_filename
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Checkpoint saved: {checkpoint_filename}")
    return checkpoint_path


def load_checkpoint(checkpoint_path: str, config: FieldDistribution) -> Tuple[int, TrainingStats, TrainingConfig]:
    """Load training progress from checkpoint file"""
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Restore training config
    training_config = TrainingConfig()
    training_config.__dict__.update(checkpoint_data['training_config'])
    
    # Restore statistics
    stats = TrainingStats()
    stats_data = checkpoint_data['stats']
    stats.episode_rewards = stats_data['episode_rewards']
    stats.episode_lengths = stats_data['episode_lengths']
    stats.win_rates = stats_data['win_rates']
    stats.goals_scored = stats_data['goals_scored']
    stats.q_table_sizes = stats_data['q_table_sizes']
    stats.exploration_rates = stats_data['exploration_rates']
    stats.collision_counts = stats_data['collision_counts']
    stats.episode_times = stats_data['episode_times']
    stats.start_time = stats_data['start_time']
    stats.best_episode_reward = stats_data['best_episode_reward']
    stats.best_win_rate = stats_data['best_win_rate']
    
    # Restore agent states
    agent_data_list = checkpoint_data['agents']
    for i, agent_data in enumerate(agent_data_list):
        if i < len(config.agents):
            agent = config.agents[i]
            
            # Restore Q-table
            agent.q_policy.Q.clear()
            for key, value in agent_data['q_table'].items():
                agent.q_policy.Q[key] = value
            
            # Restore Q-learning parameters
            params = agent_data['q_policy_params']
            agent.q_policy.alpha = params['alpha']
            agent.q_policy.gamma = params['gamma']
            agent.q_policy.eps = params['eps']
            agent.q_policy.eps_end = params['eps_end']
            agent.q_policy.eps_decay = params['eps_decay']
            agent.q_policy.episode_count = params['episode_count']
            
            # Restore episode data
            agent.episode_rewards = agent_data['episode_rewards']
            agent.cumulative_reward = agent_data['cumulative_reward']
    
    print(f"Checkpoint loaded from episode {checkpoint_data['episode']}")
    return checkpoint_data['episode'], stats, training_config


def print_training_progress(episode: int, stats: TrainingStats, config: TrainingConfig):
    """Print formatted training progress"""
    summary = stats.get_summary_stats()
    
    print(f"\n=== Episode {episode}/{config.num_episodes} ===")
    print(f"Average Reward (last 100): {summary.get('avg_reward', 0):.2f}")
    print(f"Win Rate (last 100): {summary.get('win_rate', 0):.2%}")
    print(f"Best Win Rate: {summary.get('best_win_rate', 0):.2%}")
    print(f"Average Episode Length: {summary.get('avg_episode_length', 0):.1f} steps")
    print(f"Exploration Rate: {summary.get('current_exploration_rate', 0):.3f}")
    print(f"Avg Q-table Size: {summary.get('avg_q_table_size', 0):.0f} states")
    print(f"Training Time: {summary.get('total_training_time', 0)/3600:.2f} hours")
    print(f"Avg Time/Episode: {summary.get('avg_episode_time', 0):.2f}s")


def train(config: FieldDistribution, training_config: Optional[TrainingConfig] = None,
          resume_checkpoint: Optional[str] = None) -> TrainingStats:
    """
    Train agents using Q-learning with BDI reasoning in soccer simulation.
    
    Args:
        config: Field distribution with agents to train
        training_config: Training configuration parameters
        resume_checkpoint: Path to checkpoint file to resume from
        
    Returns:
        TrainingStats object with training history
    """
    # Set default training config if not provided
    if training_config is None:
        training_config = TrainingConfig()
    
    # Initialize or load training state
    start_episode = 0
    stats = TrainingStats()
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        start_episode, stats, loaded_config = load_checkpoint(resume_checkpoint, config)
        # Use loaded config but allow override of key parameters
        training_config = loaded_config
        print(f"Resuming training from episode {start_episode}")
    else:
        print(f"Starting new training session with {len(config.agents)} agents")
        print(f"Training for {training_config.num_episodes} episodes")
        print(f"Checkpoints every {training_config.checkpoint_frequency} episodes")
    
    # Initialize environment
    env = config.agents[0].env if config.agents else Environment()
    
    # Track early stopping
    episodes_without_improvement = 0
    best_performance_episode = start_episode
    
    try:
        for episode in range(start_episode, training_config.num_episodes):
            episode_start_time = time.time()
            
            # Reset environment and agents for new episode
            observations = env.reset()
            for agent in config.agents:
                agent.reset_episode()
            
            total_episode_reward = 0
            collision_count = 0
            episode_length = 0
            
            # Episode loop
            for step in range(training_config.max_episode_steps):
                # Get actions from all agents using BDI + Q-learning
                actions = []
                for agent in config.agents:
                    action = agent.act()
                    actions.append(action)
                
                # Execute environment step
                observations, rewards, done, info = env.step(actions)
                
                # Update agents with rewards and new observations
                for agent, reward in zip(config.agents, rewards):
                    agent.learn(reward, done)
                    total_episode_reward += reward
                
                collision_count += len(info.get('collisions', []))
                episode_length = step + 1
                
                if done:
                    break
            
            episode_time = time.time() - episode_start_time
            
            # Calculate episode statistics
            avg_q_table_size = np.mean([len(agent.q_policy.Q) for agent in config.agents])
            avg_exploration_rate = np.mean([agent.q_policy.eps for agent in config.agents])
            
            # Update training statistics
            stats.update_episode_stats(
                episode_reward=total_episode_reward,
                episode_length=episode_length,
                goals=info.get('score', {'blue': 0, 'white': 0}),
                q_table_size=avg_q_table_size,
                exploration_rate=avg_exploration_rate,
                collision_count=collision_count,
                episode_time=episode_time
            )
            
            # Print progress
            if (episode + 1) % training_config.stats_frequency == 0:
                print_training_progress(episode + 1, stats, training_config)
            
            # Save checkpoint
            if (episode + 1) % training_config.checkpoint_frequency == 0:
                checkpoint_path = save_checkpoint(config, episode + 1, stats, training_config)
                
                # Check for improvement
                current_win_rate = stats.calculate_win_rate()
                if current_win_rate > stats.best_win_rate:
                    best_performance_episode = episode + 1
                    episodes_without_improvement = 0
                    print(f"New best win rate: {current_win_rate:.2%}")
                    
                    # Save best model checkpoint
                    best_checkpoint = checkpoint_path.replace('.pkl', '_BEST.pkl')
                    os.rename(checkpoint_path, best_checkpoint)
                    print(f"Best model saved: {os.path.basename(best_checkpoint)}")
                else:
                    episodes_without_improvement += training_config.checkpoint_frequency
            
            # Early stopping check
            if episodes_without_improvement >= training_config.early_stopping_patience:
                print(f"\nEarly stopping: No improvement for {episodes_without_improvement} episodes")
                break
                
            # Target performance reached
            current_win_rate = stats.calculate_win_rate()
            if current_win_rate >= training_config.target_win_rate:
                print(f"\nTarget win rate {training_config.target_win_rate:.2%} achieved!")
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save emergency checkpoint
        emergency_checkpoint = save_checkpoint(config, episode, stats, training_config)
        print(f"Emergency checkpoint saved: {emergency_checkpoint}")
        
    # Final checkpoint and summary
    final_checkpoint = save_checkpoint(config, episode + 1, stats, training_config)
    print(f"\nTraining completed!")
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"Best performance at episode {best_performance_episode}")
    
    # Print final statistics
    final_stats = stats.get_summary_stats()
    print("\n=== Final Training Summary ===")
    print(f"Episodes completed: {final_stats['episodes_completed']}")
    print(f"Best win rate achieved: {final_stats['best_win_rate']:.2%}")
    print(f"Final average reward: {final_stats['avg_reward']:.2f}")
    print(f"Total training time: {final_stats['total_training_time']/3600:.2f} hours")
    
    return stats

def main():
    """
    Main function demonstrating training setup and execution.
    """
    # Create environment
    env = Environment()
    
    # Create agents for training (field center at 0,0)
    # Blue team (attacking left to right, starts on left side)
    attacker_blue = Attacker(env, Team.BLUE, pos=np.array([-env.width * 0.25, -env.height * 0.1]))
    defender_blue = Defender(env, Team.BLUE, pos=np.array([-env.width * 0.4, env.height * 0.1]))
    
    # White team (attacking right to left, starts on right side)  
    attacker_white = Attacker(env, Team.WHITE, pos=np.array([env.width * 0.25, env.height * 0.1]))
    defender_white = Defender(env, Team.WHITE, pos=np.array([env.width * 0.4, -env.height * 0.1]))
    
    # Create field distribution with all agents
    config = FieldDistribution()
    config.add(attacker_blue)
    config.add(defender_blue)
    config.add(attacker_white)
    config.add(defender_white)
    
    # Configure training parameters
    training_config = TrainingConfig(
        num_episodes=2000,           # Train for 2000 episodes
        checkpoint_frequency=50,     # Save checkpoint every 50 episodes
        stats_frequency=10,          # Print stats every 10 episodes
        save_directory="checkpoints",
        max_episode_steps=600,       # 30 seconds at 20 FPS
        learning_rate=0.1,
        gamma=0.95,
        eps_start=1.0,              # Start with full exploration
        eps_end=0.05,               # End with 5% exploration
        eps_decay=0.995,            # Decay rate per episode
        early_stopping_patience=300, # Stop if no improvement for 300 episodes
        target_win_rate=0.75        # Target 75% win rate
    )
    
    print("Soccer Robocup Multi-Agent Q-Learning with BDI")
    print(f"Training {len(config.agents)} agents ({[a.role for a in config.agents]})")
    print(f"Environment: {env.width}x{env.height} field")
    print(f"Episodes: {training_config.num_episodes}")
    print(f"Checkpoint frequency: {training_config.checkpoint_frequency}")
    print("\nStarting training...\n")
    
    # Start training
    try:
        # Check if we want to resume from a checkpoint
        resume_path = None
        if os.path.exists(training_config.save_directory):
            checkpoints = [f for f in os.listdir(training_config.save_directory) 
                         if f.endswith('.pkl') and 'checkpoint' in f]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(
                    os.path.join(training_config.save_directory, x)
                ))
                resume_path = os.path.join(training_config.save_directory, latest_checkpoint)
                print(f"Found existing checkpoint: {latest_checkpoint}")
                
                # Ask user if they want to resume (simplified - just resume automatically)
                print(f"Resuming from: {resume_path}")
        
        # Execute training
        final_stats = train(config, training_config, resume_checkpoint=resume_path)
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Total episodes: {len(final_stats.episode_rewards)}")
        print(f"Best win rate: {final_stats.best_win_rate:.2%}")
        print(f"Checkpoints saved in: {training_config.save_directory}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
