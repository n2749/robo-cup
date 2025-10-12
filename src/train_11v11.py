#!/usr/bin/env python3
"""
11v11 Soccer Training Script
Full team soccer simulation with realistic formations and team play.
"""

import argparse

from main import TrainingConfig, train
from team_config import (
    create_11v11_field_distribution, 
    get_available_formations,
    print_formation_info
)
import numpy as np


def create_11v11_training_config(
    num_episodes: int = 5000,
    checkpoint_frequency: int = 100,
    stats_frequency: int = 20,
    learning_rate: float = 0.05,  # Lower for more stability with many agents
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.02,  # Lower final exploration for better coordination
    eps_decay: float = 0.999,  # Slower decay for longer exploration
    early_stopping_patience: int = 500,
    target_win_rate: float = 0.8,
    max_episode_steps: int = 1200  # Longer episodes for full teams
) -> TrainingConfig:
    """
    Create optimized training configuration for 11v11 matches.
    
    The configuration is tuned for:
    - 22 agents learning simultaneously
    - More complex tactical play
    - Longer episodes to allow tactical development
    - More stable learning with lower learning rates
    """
    
    return TrainingConfig(
        num_episodes=num_episodes,
        checkpoint_frequency=checkpoint_frequency,
        stats_frequency=stats_frequency,
        save_directory="checkpoints/11v11",
        max_episode_steps=max_episode_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        early_stopping_patience=early_stopping_patience,
        target_win_rate=target_win_rate
    )


def main():
    parser = argparse.ArgumentParser(description='11v11 Soccer Training')
    parser.add_argument('--blue-formation', '-b', type=str, default='4-4-2',
                       help='Formation for blue team')
    parser.add_argument('--white-formation', '-w', type=str, default='4-4-2', 
                       help='Formation for white team')
    parser.add_argument('--episodes', '-e', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--checkpoint-freq', '-c', type=int, default=100,
                       help='Checkpoint frequency')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.05,
                       help='Learning rate')
    parser.add_argument('--list-formations', '-lf', action='store_true',
                       help='List available formations')
    parser.add_argument('--resume', '-r', type=str,
                       help='Resume from checkpoint file')
    parser.add_argument('--quick-test', '-q', action='store_true',
                       help='Quick test with fewer episodes')
    
    args = parser.parse_args()
    
    # List formations if requested
    if args.list_formations:
        print_formation_info()
        return
    
    # Get available formations
    formations = get_available_formations()
    
    # Validate formations
    if args.blue_formation not in formations:
        print(f"Error: Unknown blue formation '{args.blue_formation}'")
        print("Available formations:", list(formations.keys()))
        return
    
    if args.white_formation not in formations:
        print(f"Error: Unknown white formation '{args.white_formation}'")
        print("Available formations:", list(formations.keys()))
        return
    
    # Quick test configuration
    if args.quick_test:
        args.episodes = 50
        args.checkpoint_freq = 10
        print("Quick test mode: 50 episodes, checkpoint every 10")
    
    print("=" * 70)
    print("11v11 SOCCER SIMULATION TRAINING")
    print("=" * 70)
    print(f"Blue team formation: {args.blue_formation}")
    print(f"White team formation: {args.white_formation}")
    print(f"Training episodes: {args.episodes}")
    print(f"Checkpoint frequency: {args.checkpoint_freq}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Create team configurations
    blue_formation = formations[args.blue_formation]
    white_formation = formations[args.white_formation]
    
    print(f"\nSetting up teams...")
    config = create_11v11_field_distribution(blue_formation, white_formation)
    
    # Create training configuration
    training_config = create_11v11_training_config(
        num_episodes=args.episodes,
        checkpoint_frequency=args.checkpoint_freq,
        learning_rate=args.learning_rate
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {training_config.num_episodes}")
    print(f"  Max episode steps: {training_config.max_episode_steps} ({training_config.max_episode_steps * 0.05:.1f}s)")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Exploration: {training_config.eps_start} → {training_config.eps_end}")
    print(f"  Target win rate: {training_config.target_win_rate:.1%}")
    print(f"  Early stopping patience: {training_config.early_stopping_patience}")
    print(f"  Checkpoint directory: {training_config.save_directory}")
    
    # Team statistics
    role_counts_blue = {}
    role_counts_white = {}
    
    for agent in config.agents:
        if agent.team.name == 'BLUE':
            role_counts_blue[agent.role] = role_counts_blue.get(agent.role, 0) + 1
        else:
            role_counts_white[agent.role] = role_counts_white.get(agent.role, 0) + 1
    
    print(f"\nTeam Composition:")
    print(f"  Blue team ({blue_formation.name}):")
    for role, count in sorted(role_counts_blue.items()):
        print(f"    {role}: {count}")
    print(f"  White team ({white_formation.name}):")
    for role, count in sorted(role_counts_white.items()):
        print(f"    {role}: {count}")
    
    print(f"\nTotal agents: {len(config.agents)}")
    print(f"Expected Q-table size per agent: ~500-2000 states")
    print(f"Total Q-table entries: ~{len(config.agents) * 1000:,}")
    
    # Memory and performance estimates
    estimated_memory = len(config.agents) * 1000 * 8 / (1024 * 1024)  # Rough MB estimate
    print(f"Estimated memory usage: ~{estimated_memory:.1f} MB")
    
    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING...")
    print("=" * 70)
    print("\nTips for 11v11 training:")
    print("- Training will be slower due to 22 agents")
    print("- Tactical coordination may take 1000+ episodes to develop")
    print("- Watch for role-specific Q-table growth in checkpoints")
    print("- Early episodes will appear chaotic - this is normal")
    print("\nPress Ctrl+C to stop and save emergency checkpoint")
    print()
    
    try:
        # Execute training
        final_stats = train(
            config, 
            training_config, 
            resume_checkpoint=args.resume
        )
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Final statistics
        summary = final_stats.get_summary_stats()
        print(f"\nFinal Results:")
        print(f"  Episodes completed: {summary['episodes_completed']}")
        print(f"  Best win rate: {summary['best_win_rate']:.2%}")
        print(f"  Final average reward: {summary['avg_reward']:.2f}")
        print(f"  Total training time: {summary['total_training_time']/3600:.2f} hours")
        print(f"  Average Q-table size: {summary['avg_q_table_size']:.0f} states per agent")
        
        # Team performance breakdown
        blue_agents = [a for a in config.agents if a.team.name == 'BLUE']
        white_agents = [a for a in config.agents if a.team.name == 'WHITE']
        
        blue_avg_reward = np.mean([np.mean(a.episode_rewards[-10:]) if a.episode_rewards else 0 for a in blue_agents])
        white_avg_reward = np.mean([np.mean(a.episode_rewards[-10:]) if a.episode_rewards else 0 for a in white_agents])
        
        print(f"\nTeam Performance (last 10 episodes):")
        print(f"  Blue team average reward: {blue_avg_reward:.2f}")
        print(f"  White team average reward: {white_avg_reward:.2f}")
        
        # Role-specific performance
        print(f"\nRole-specific Q-table sizes:")
        role_qtable_sizes = {}
        for agent in config.agents:
            role = agent.role
            if role not in role_qtable_sizes:
                role_qtable_sizes[role] = []
            role_qtable_sizes[role].append(len(agent.q_policy.Q))
        
        for role, sizes in role_qtable_sizes.items():
            avg_size = np.mean(sizes)
            print(f"  {role}: {avg_size:.0f} states (avg)")
        
        print(f"\nCheckpoints saved in: {training_config.save_directory}")
        print(f"Use visualize.py to watch the trained teams play!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Emergency checkpoint should have been saved")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
