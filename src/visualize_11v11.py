#!/usr/bin/env python3
"""
11v11 Soccer Visualization Script
Enhanced visualization for full team soccer matches with formations display.
"""

import os
import argparse
import pygame
from team_config import get_available_formations
import pickle
import numpy as np
from typing import List, Dict, Tuple


def list_11v11_checkpoints(checkpoint_dir: str = "checkpoints_11v11") -> List[str]:
    """List available 11v11 checkpoint files."""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pkl') and 'checkpoint' in filename:
            checkpoints.append(os.path.join(checkpoint_dir, filename))
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoints


def analyze_checkpoint(checkpoint_path: str) -> Dict:
    """Analyze a checkpoint file and return team information."""
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        episode = checkpoint_data['episode']
        agents_data = checkpoint_data['agents']
        stats_data = checkpoint_data['stats']
        
        # Count team composition
        blue_roles = {}
        white_roles = {}
        
        for agent_data in agents_data:
            role = agent_data['role']
            team = agent_data['team']
            
            if team == 'BLUE':
                blue_roles[role] = blue_roles.get(role, 0) + 1
            else:
                white_roles[role] = white_roles.get(role, 0) + 1
        
        # Calculate statistics
        total_episodes = len(stats_data['episode_rewards'])
        avg_reward = np.mean(stats_data['episode_rewards'][-100:]) if stats_data['episode_rewards'] else 0
        
        recent_blue = stats_data['goals_scored']['blue'][-100:] if stats_data['goals_scored']['blue'] else []
        recent_white = stats_data['goals_scored']['white'][-100:] if stats_data['goals_scored']['white'] else []
        
        if len(recent_blue) > 0 and len(recent_white) > 0:
            wins = sum(1 for b, w in zip(recent_blue, recent_white) if b > w)
            win_rate = wins / len(recent_blue)
        else:
            win_rate = 0.0
        
        return {
            'episode': episode,
            'total_episodes': total_episodes,
            'blue_team': blue_roles,
            'white_team': white_roles,
            'avg_reward': avg_reward,
            'win_rate': win_rate,
            'valid': True
        }
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def print_checkpoint_analysis(checkpoint_path: str):
    """Print detailed analysis of a checkpoint."""
    analysis = analyze_checkpoint(checkpoint_path)
    
    if not analysis['valid']:
        print(f"Error analyzing checkpoint: {analysis['error']}")
        return
    
    print(f"Checkpoint Analysis: {os.path.basename(checkpoint_path)}")
    print("=" * 60)
    print(f"Episode: {analysis['episode']}")
    print(f"Total Episodes Completed: {analysis['total_episodes']}")
    print(f"Average Reward (last 100): {analysis['avg_reward']:.2f}")
    print(f"Win Rate (last 100): {analysis['win_rate']:.2%}")
    
    print(f"\nBlue Team Composition:")
    for role, count in sorted(analysis['blue_team'].items()):
        print(f"  {role}: {count}")
    
    print(f"\nWhite Team Composition:")
    for role, count in sorted(analysis['white_team'].items()):
        print(f"  {role}: {count}")
    
    total_agents = sum(analysis['blue_team'].values()) + sum(analysis['white_team'].values())
    print(f"\nTotal Agents: {total_agents}")


def run_pygame_visualization(checkpoint_path: str):
    
    # Import visualization after confirming pygame is available
    from visualize import SoccerVisualizer
    
    print(f"Starting pygame visualization...")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    
    visualizer = SoccerVisualizer(width=1400, height=900)  # Larger for 11v11
    
    try:
        if visualizer.load_checkpoint_and_setup(checkpoint_path):
            print("\nVisualization Controls:")
            print("  SPACE - Pause/Resume")
            print("  +/-   - Adjust speed")
            print("  T     - Toggle trails")
            print("  B     - Toggle beliefs")
            print("  S     - Toggle statistics") 
            print("  Q     - Quit")
            print("\nStarting visualization...")
            
            visualizer.run_continuous_visualization()
            return True
        else:
            print("Failed to load checkpoint for visualization.")
            return False
    
    except KeyboardInterrupt:
        print("\nVisualization interrupted.")
        return True
    
    finally:
        visualizer.close()


def run_text_visualization(checkpoint_path: str, num_episodes: int = 3):
    """Run text-based visualization showing match results."""
    print(f"Running text-based match simulation...")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    
    analysis = analyze_checkpoint(checkpoint_path)
    if not analysis['valid']:
        print(f"Cannot load checkpoint: {analysis['error']}")
        return
    
    print(f"\nMatch Setup:")
    print(f"Blue Team: {', '.join([f'{count} {role}' for role, count in analysis['blue_team'].items()])}")
    print(f"White Team: {', '.join([f'{count} {role}' for role, count in analysis['white_team'].items()])}")
    print(f"Training Level: Episode {analysis['episode']}")
    print(f"Team Performance: {analysis['win_rate']:.1%} win rate")
    print()
    
    # Simulate some matches based on checkpoint data
    print(f"Simulating {num_episodes} matches...")
    print("-" * 50)
    
    for i in range(num_episodes):
        # Simple simulation based on team strength (win rate)
        outcome = np.random.random()
        
        if outcome < analysis['win_rate']:
            blue_score = np.random.randint(1, 4)
            white_score = np.random.randint(0, blue_score)
        else:
            white_score = np.random.randint(1, 4)
            blue_score = np.random.randint(0, white_score)
        
        print(f"Match {i+1}: Blue {blue_score} - {white_score} White")
    
    print("-" * 50)
    print("Simulation complete!")


def main():
    parser = argparse.ArgumentParser(description='11v11 Soccer Visualization')
    parser.add_argument('--checkpoint', '-c', type=str, 
                       help='Path to specific checkpoint file')
    parser.add_argument('--checkpoint-dir', '-d', type=str, default='checkpoints_11v11',
                       help='Directory containing 11v11 checkpoints')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available checkpoints with analysis')
    parser.add_argument('--analyze', '-a', type=str,
                       help='Analyze specific checkpoint file')
    parser.add_argument('--text-sim', '-t', action='store_true',
                       help='Run text-based simulation instead of pygame')
    parser.add_argument('--formations', '-f', action='store_true',
                       help='Show available formations')
    
    args = parser.parse_args()
    
    # Show formations if requested
    if args.formations:
        from team_config import print_formation_info
        print_formation_info()
        return
    
    # List checkpoints if requested
    if args.list:
        checkpoints = list_11v11_checkpoints(args.checkpoint_dir)
        if checkpoints:
            print("Available 11v11 checkpoints:\n")
            for i, checkpoint in enumerate(checkpoints[:10]):  # Show top 10
                print(f"{i+1}. {os.path.basename(checkpoint)}")
                analysis = analyze_checkpoint(checkpoint)
                if analysis['valid']:
                    print(f"   Episode: {analysis['episode']}, "
                          f"Win Rate: {analysis['win_rate']:.1%}, "
                          f"Agents: {sum(analysis['blue_team'].values()) + sum(analysis['white_team'].values())}")
                print()
        else:
            print(f"No checkpoints found in {args.checkpoint_dir}")
            print("Run train_11v11.py first to create 11v11 training data")
        return
    
    # Analyze specific checkpoint
    if args.analyze:
        if os.path.exists(args.analyze):
            print_checkpoint_analysis(args.analyze)
        else:
            print(f"Checkpoint file not found: {args.analyze}")
        return
    
    # Get checkpoint to visualize
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoints = list_11v11_checkpoints(args.checkpoint_dir)
        if checkpoints:
            checkpoint_path = checkpoints[0]  # Use most recent
            print(f"Using most recent checkpoint: {os.path.basename(checkpoint_path)}")
        else:
            print(f"No checkpoints found in {args.checkpoint_dir}")
            print("Run train_11v11.py first to create 11v11 training data")
            return
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return
    
    # Show checkpoint analysis first
    print_checkpoint_analysis(checkpoint_path)
    print()
    
    # Run visualization
    success = run_pygame_visualization(checkpoint_path)
    if not success:
        print("Pygame visualization failed, falling back to text simulation...")
        run_text_visualization(checkpoint_path)


if __name__ == "__main__":
    main()
