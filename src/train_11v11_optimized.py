#!/usr/bin/env python3
"""
Optimized 11v11 Soccer Training System

This script is specifically optimized for full 11v11 soccer training with:
- Parallel processing for maximum CPU utilization
- Proper team formations and role assignments
- Efficient checkpointing and progress tracking
- Performance monitoring and optimization
- Smart episode management and early stopping
"""

import sys
import os
sys.path.append('src')

import time
import numpy as np
from typing import Dict, List, Optional
import multiprocessing as mp

try:
    from performance_utils import ParallelTrainer
    from main import TrainingConfig
    from agents import FieldDistribution, Agent, Team, Defender, Attacker, Goalkeeper
    from env import Environment
    from team_config import create_11v11_field_distribution, Formation442, Formation433, Formation352
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the robo-cup directory and all dependencies are available")
    sys.exit(1)


class OptimizedEleven:
    """Optimized 11v11 training system with advanced performance features."""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.optimal_processes = max(2, min(self.cpu_count - 1, 8))  # Leave 1 core free, cap at 8
        print(f"Detected {self.cpu_count} CPU cores, using {self.optimal_processes} for parallel training")
    
    def create_11v11_teams(self, formation_blue: str = "4-4-2", formation_white: str = "4-4-2") -> FieldDistribution:
        """
        Create optimized 11v11 teams with proper formations.
        
        Args:
            formation_blue: Formation for blue team (4-4-2, 4-3-3, 3-5-2)
            formation_white: Formation for white team
            
        Returns:
            FieldDistribution with 22 properly positioned agents
        """
        print(f"Creating 11v11 teams: Blue ({formation_blue}) vs White ({formation_white})")
        
        env = Environment()
        config = FieldDistribution()
        
        try:
            # Map formation strings to formation classes
            formation_map = {
                "4-4-2": Formation442(),
                "4-3-3": Formation433(), 
                "3-5-2": Formation352()
            }
            
            blue_form = formation_map.get(formation_blue, Formation442())
            white_form = formation_map.get(formation_white, Formation442())
            
            # Create field distribution with formations
            config = create_11v11_field_distribution(blue_form, white_form)
                
        except Exception as e:
            print(f"Error creating teams with formations: {e}")
            print("Falling back to basic team creation...")
            config = self._create_basic_11v11_teams(env)
        
        print(f"Created {len(config.agents)} agents total")
        self._print_team_composition(config)
        
        return config
    
    def _create_basic_11v11_teams(self, env: Environment) -> FieldDistribution:
        """Fallback method to create basic 11v11 teams."""
        config = FieldDistribution()
        
        formations = {
            # Blue team (attacks right, defends left goal)
            Team.BLUE: {
                'goalkeeper': [(np.array([-45, 0]), 'goalkeeper')],
                'defenders': [
                    (np.array([-35, -15]), 'defender'),
                    (np.array([-35, -5]), 'defender'), 
                    (np.array([-35, 5]), 'defender'),
                    (np.array([-35, 15]), 'defender')
                ],
                'midfielders': [
                    (np.array([-15, -20]), 'midfielder'),
                    (np.array([-15, -7]), 'midfielder'),
                    (np.array([-15, 0]), 'midfielder'),
                    (np.array([-15, 7]), 'midfielder'),
                    (np.array([-15, 20]), 'midfielder')
                ],
                'attackers': [
                    (np.array([5, -10]), 'attacker'),
                    (np.array([5, 10]), 'attacker')
                ]
            },
            # White team (attacks left, defends right goal)  
            Team.WHITE: {
                'goalkeeper': [(np.array([45, 0]), 'goalkeeper')],
                'defenders': [
                    (np.array([35, -15]), 'defender'),
                    (np.array([35, -5]), 'defender'),
                    (np.array([35, 5]), 'defender'), 
                    (np.array([35, 15]), 'defender')
                ],
                'midfielders': [
                    (np.array([15, -20]), 'midfielder'),
                    (np.array([15, -7]), 'midfielder'),
                    (np.array([15, 0]), 'midfielder'),
                    (np.array([15, 7]), 'midfielder'),
                    (np.array([15, 20]), 'midfielder')
                ],
                'attackers': [
                    (np.array([-5, -10]), 'attacker'),
                    (np.array([-5, 10]), 'attacker')
                ]
            }
        }
        
        # Create agents for both teams
        for team, positions in formations.items():
            for role_group, role_positions in positions.items():
                for pos, role in role_positions:
                    if role == 'goalkeeper':
                        agent = Goalkeeper(env, team, pos=pos)
                    elif role == 'defender':
                        agent = Defender(env, team, pos=pos)
                    elif role == 'attacker':
                        agent = Attacker(env, team, pos=pos)
                    else:  # midfielder
                        agent = Agent(env, team, role='midfielder', pos=pos)
                    
                    config.add(agent)
        
        return config
    
    def _print_team_composition(self, config: FieldDistribution):
        """Print team composition for verification."""
        blue_roles = {}
        white_roles = {}
        
        for agent in config.agents:
            role = agent.role
            if agent.team == Team.BLUE:
                blue_roles[role] = blue_roles.get(role, 0) + 1
            else:
                white_roles[role] = white_roles.get(role, 0) + 1
        
        print("\nTeam Composition:")
        print(f"Blue Team: {dict(blue_roles)}")
        print(f"White Team: {dict(white_roles)}")
    
    def create_optimized_training_config(self, episodes: int = 1000, quick_test: bool = False) -> TrainingConfig:
        """
        Create optimized training configuration for 11v11.
        
        Args:
            episodes: Number of episodes to train
            quick_test: If True, use settings for quick testing
            
        Returns:
            Optimized TrainingConfig
        """
        if quick_test:
            return TrainingConfig(
                num_episodes=50,
                checkpoint_frequency=10,
                stats_frequency=5,
                max_episode_steps=200,  # Shorter episodes for testing
                early_stopping_patience=20,
                target_win_rate=0.6,
                learning_rate=0.15,  # Slightly higher for faster learning
                eps_decay=0.99  # Faster exploration decay
            )
        else:
            return TrainingConfig(
                num_episodes=episodes,
                checkpoint_frequency=max(25, episodes // 40),  # ~40 checkpoints max
                stats_frequency=max(10, episodes // 100),      # ~100 progress updates max
                max_episode_steps=400,                         # Longer episodes for full games
                early_stopping_patience=episodes // 10,       # Stop if no improvement
                target_win_rate=0.75,                         # Good performance target
                learning_rate=0.1,                            # Standard learning rate
                gamma=0.95,                                    # Standard discount
                eps_start=1.0,
                eps_end=0.05,
                eps_decay=0.995,                               # Gradual exploration decay
                save_directory="checkpoints/11v11/"            # Dedicated directory
            )
    
    def estimate_training_time(self, episodes: int) -> Dict[str, float]:
        """Estimate training time for different configurations."""
        
        # Base estimates from performance analysis
        base_episode_time = 2.5  # seconds per episode for 6v6
        scaling_factor = (22 / 12) ** 1.3  # Slightly super-linear scaling for 11v11
        
        single_episode_time = base_episode_time * scaling_factor
        
        estimates = {
            'single_core_hours': (single_episode_time * episodes) / 3600,
            'parallel_2_hours': (single_episode_time * episodes) / (3600 * 1.75),
            'parallel_4_hours': (single_episode_time * episodes) / (3600 * 2.54),
            'optimal_parallel_hours': (single_episode_time * episodes) / (3600 * min(3.0, self.optimal_processes * 0.7)),
            'episodes': episodes,
            'estimated_episode_time': single_episode_time
        }
        
        return estimates
    
    def print_training_estimates(self, episodes: int):
        """Print training time estimates."""
        estimates = self.estimate_training_time(episodes)
        
        print(f"\n🕒 Training Time Estimates for {episodes} episodes:")
        print(f"  Single-threaded: {estimates['single_core_hours']:.2f} hours")
        print(f"  2-core parallel:  {estimates['parallel_2_hours']:.2f} hours")
        print(f"  4-core parallel:  {estimates['parallel_4_hours']:.2f} hours")
        print(f"  Optimal parallel: {estimates['optimal_parallel_hours']:.2f} hours ({self.optimal_processes} cores)")
        print(f"  Est. episode time: {estimates['estimated_episode_time']:.2f}s")
    
    def run_optimized_11v11_training(
        self,
        episodes: int = 1000,
        formation_blue: str = "4-4-2",
        formation_white: str = "4-4-2", 
        use_parallel: bool = True,
        num_processes: Optional[int] = None,
        quick_test: bool = False
    ) -> Dict:
        """
        Run optimized 11v11 training with all performance enhancements.
        
        Args:
            episodes: Number of episodes to train
            formation_blue: Blue team formation
            formation_white: White team formation
            use_parallel: Whether to use parallel processing
            num_processes: Number of processes (None for auto-detect)
            quick_test: If True, run quick test configuration
            
        Returns:
            Training results and statistics
        """
        
        print("🏆 OPTIMIZED 11v11 SOCCER TRAINING")
        print("=" * 50)
        
        # Print estimates
        self.print_training_estimates(episodes if not quick_test else 50)
        
        # Create teams
        start_setup = time.time()
        config = self.create_11v11_teams(formation_blue, formation_white)
        setup_time = time.time() - start_setup
        print(f"\nTeam setup completed in {setup_time:.2f}s")
        
        # Create training configuration
        training_config = self.create_optimized_training_config(episodes, quick_test)
        
        # Determine process count
        if num_processes is None:
            num_processes = self.optimal_processes if use_parallel else 1
        
        print(f"\nTraining Configuration:")
        print(f"  Episodes: {training_config.num_episodes}")
        print(f"  Max episode steps: {training_config.max_episode_steps}")
        print(f"  Parallel processing: {use_parallel} ({num_processes} processes)")
        print(f"  Early stopping: {training_config.early_stopping_patience} episodes")
        print(f"  Target win rate: {training_config.target_win_rate:.0%}")
        
        # Confirm before long training
        if not quick_test and episodes >= 500:
            response = input(f"\nStart training {episodes} episodes? This may take several hours. [y/N]: ")
            if response.lower() != 'y':
                print("Training cancelled.")
                return {'cancelled': True}
        
        print(f"\n🚀 Starting training...")
        start_time = time.time()
        
        try:
            if use_parallel:
                # Use parallel training
                trainer = ParallelTrainer(num_processes=num_processes)
                results = trainer.parallel_train(config, training_config)
                
                print(f"\n✅ Parallel training completed!")
                print(f"  Speedup factor: {results.get('speedup_factor', 'N/A'):.2f}x")
                print(f"  Workers used: {results.get('workers_used', 'N/A')}")
                print(f"  Episodes completed: {results.get('total_episodes_completed', 'N/A')}")
                
            else:
                # Regular single-threaded training
                from main import train
                results = train(config, training_config)
                print(f"\n✅ Single-threaded training completed!")
            
            total_time = time.time() - start_time
            
            # Training summary
            print(f"\n🎯 TRAINING SUMMARY")
            print(f"  Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
            print(f"  Configuration: 11v11 ({formation_blue} vs {formation_white})")
            print(f"  Episodes: {training_config.num_episodes}")
            
            if use_parallel:
                actual_speedup = (training_config.num_episodes * 4.5) / total_time  # Rough estimate
                print(f"  Effective speedup: ~{actual_speedup:.1f}x")
            
            # Performance metrics
            if isinstance(results, dict) and 'worker_results' in results:
                all_rewards = []
                for worker_results in results['worker_results'].values():
                    if 'avg_reward' in worker_results:
                        all_rewards.append(worker_results['avg_reward'])
                
                if all_rewards:
                    print(f"  Average reward across workers: {np.mean(all_rewards):.1f}")
                    print(f"  Reward std dev: {np.std(all_rewards):.1f}")
            
            results['training_time_hours'] = total_time / 3600
            results['training_config'] = training_config.__dict__
            results['team_config'] = {
                'formation_blue': formation_blue,
                'formation_white': formation_white,
                'total_agents': len(config.agents)
            }
            
            return results
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user after {(time.time() - start_time)/60:.1f} minutes")
            return {'interrupted': True, 'partial_time': (time.time() - start_time) / 3600}
        
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


def run_quick_test():
    """Run a quick test of the 11v11 system."""
    print("🧪 Running quick 11v11 test...")
    
    optimizer = OptimizedEleven()
    results = optimizer.run_optimized_11v11_training(
        episodes=20,  # Very short test
        quick_test=True,
        use_parallel=True
    )
    
    if results.get('error'):
        print(f"❌ Test failed: {results['error']}")
        return False
    elif results.get('interrupted') or results.get('cancelled'):
        print("⚠️  Test was interrupted")
        return False
    else:
        print("✅ Quick test completed successfully!")
        return True


def main():
    """Main function with user interaction for 11v11 training."""
    
    print("🏆 OPTIMIZED 11v11 SOCCER TRAINING SYSTEM")
    print("=" * 60)
    
    optimizer = OptimizedEleven()
    
    print("\nAvailable options:")
    print("1. Quick test (20 episodes, ~2-5 minutes)")
    print("2. Medium training (200 episodes, ~20-40 minutes)")
    print("3. Full training (1000 episodes, ~2-4 hours)")
    print("4. Custom training")
    print("5. Performance analysis only")
    
    try:
        choice = input("\nSelect option [1-5]: ").strip()
        
        if choice == '1':
            return run_quick_test()
            
        elif choice == '2':
            results = optimizer.run_optimized_11v11_training(
                episodes=200,
                formation_blue="4-4-2",
                formation_white="4-3-3",
                use_parallel=True
            )
            
        elif choice == '3':
            results = optimizer.run_optimized_11v11_training(
                episodes=1000,
                formation_blue="4-4-2", 
                formation_white="4-4-2",
                use_parallel=True
            )
            
        elif choice == '4':
            # Custom training
            episodes = int(input("Number of episodes [1000]: ") or "1000")
            blue_formation = input("Blue team formation [4-4-2]: ") or "4-4-2"
            white_formation = input("White team formation [4-4-2]: ") or "4-4-2"
            parallel = input("Use parallel processing [Y/n]: ").lower() != 'n'
            
            results = optimizer.run_optimized_11v11_training(
                episodes=episodes,
                formation_blue=blue_formation,
                formation_white=white_formation,
                use_parallel=parallel
            )
            
        elif choice == '5':
            # Performance analysis only
            episodes = int(input("Analyze episodes [1000]: ") or "1000")
            optimizer.print_training_estimates(episodes)
            return True
            
        else:
            print("Invalid option selected")
            return False
        
        # Print final results
        if isinstance(results, dict) and not results.get('error') and not results.get('cancelled'):
            print(f"\n🎉 Training completed successfully!")
            
            if 'training_time_hours' in results:
                time_hours = results['training_time_hours']
                if time_hours < 1:
                    print(f"Training time: {time_hours * 60:.1f} minutes")
                else:
                    print(f"Training time: {time_hours:.2f} hours")
            
            print(f"Checkpoints saved in: checkpoints/11v11/")
            print(f"Use visualization tools to replay training results!")
            
        return True
        
    except KeyboardInterrupt:
        print(f"\n👋 Goodbye!")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
