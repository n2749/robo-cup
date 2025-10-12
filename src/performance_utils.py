"""
Performance optimization utilities for soccer training.
"""

import time
import numpy as np
import multiprocessing as mp
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os

from main import train, FieldDistribution, TrainingConfig
from agents import Agent, Team
from env import Environment


class PerformanceProfiler:
    """Profile training performance and identify bottlenecks."""
    
    def __init__(self):
        self.timings = {}
        self.counters = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing a section."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str):
        """End timing a section."""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            del self.start_times[name]
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter."""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        # Timing stats
        for name, times in self.timings.items():
            stats[f"{name}_total"] = sum(times)
            stats[f"{name}_avg"] = np.mean(times)
            stats[f"{name}_count"] = len(times)
        
        # Counter stats
        stats.update(self.counters)
        
        return stats
    
    def print_stats(self):
        """Print performance statistics."""
        stats = self.get_stats()
        
        print("\n=== Performance Profile ===")
        
        # Sort by total time
        timing_items = [(k, v) for k, v in stats.items() if k.endswith('_total')]
        timing_items.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTime spent by category:")
        for name, total_time in timing_items:
            base_name = name.replace('_total', '')
            avg_time = stats.get(f"{base_name}_avg", 0)
            count = stats.get(f"{base_name}_count", 0)
            print(f"  {base_name:20} | Total: {total_time:.4f}s | Avg: {avg_time:.6f}s | Count: {count}")
        
        print("\nCounters:")
        counter_items = [(k, v) for k, v in stats.items() if not k.endswith(('_total', '_avg', '_count'))]
        for name, value in sorted(counter_items):
            print(f"  {name:20} | {value}")


class OptimizedQLearningCache:
    """Optimized Q-learning with caching for better performance."""
    
    def __init__(self, cache_size: int = 10000):
        self.state_cache = {}
        self.cache_size = cache_size
        
    @lru_cache(maxsize=1000)
    def cached_state_key(self, distance_ball: int, distance_goal: int, 
                        distance_home: int, distance_opp: int, 
                        teammate_open: bool, goal_open: bool) -> Tuple:
        """Cached state key generation."""
        return (distance_ball, distance_goal, distance_home, 
                distance_opp, teammate_open, goal_open)
    
    def get_discretized_state(self, beliefs) -> Tuple:
        """Get discretized state with caching."""
        # Use cached discretization
        return self.cached_state_key(
            min(4, int(beliefs.distance_to_ball / 10.0)) if beliefs.distance_to_ball is not None else 0,
            min(4, int(beliefs.distance_to_goal / 20.0)) if beliefs.distance_to_goal is not None else 0,
            min(4, int(beliefs.distance_to_home_goal / 20.0)) if beliefs.distance_to_home_goal is not None else 0,
            min(4, int(beliefs.distance_to_opponent / 10.0)) if beliefs.distance_to_opponent is not None else 0,
            bool(beliefs.teammate_open) if beliefs.teammate_open is not None else False,
            bool(beliefs.goal_open) if beliefs.goal_open is not None else False
        )


class ParallelTrainer:
    """Train multiple environments in parallel for better performance."""
    
    def __init__(self, num_processes: Optional[int] = None):
        self.num_processes = num_processes or max(1, mp.cpu_count() - 1)
        print(f"Initializing parallel trainer with {self.num_processes} processes")
    
    def _train_worker(self, args: Tuple) -> Tuple[Dict, str]:
        """Worker function for parallel training."""
        config, training_config, worker_id, episodes_per_worker = args
        
        # Modify training config for this worker
        worker_config = TrainingConfig(
            num_episodes=episodes_per_worker,
            checkpoint_frequency=max(1, episodes_per_worker // 2),
            stats_frequency=max(1, episodes_per_worker // 10),
            save_directory=os.path.join(training_config.save_directory, f"worker_{worker_id}"),
            max_episode_steps=training_config.max_episode_steps,
            learning_rate=training_config.learning_rate,
            gamma=training_config.gamma,
            eps_start=training_config.eps_start,
            eps_end=training_config.eps_end,
            eps_decay=training_config.eps_decay,
            early_stopping_patience=training_config.early_stopping_patience,
            target_win_rate=training_config.target_win_rate
        )
        
        # Create independent environment for this worker
        env = Environment()
        worker_field_config = FieldDistribution()
        
        # Clone agents for this worker
        for agent in config.agents:
            new_agent = Agent(env, agent.team, role=agent.role, pos=agent.pos.copy())
            worker_field_config.add(new_agent)
        
        # Train in this process
        stats = train(worker_field_config, worker_config)
        
        return stats.get_summary_stats(), f"worker_{worker_id}"
    
    def parallel_train(self, config: FieldDistribution, training_config: TrainingConfig) -> Dict:
        """Train using multiple parallel processes."""
        
        # Divide episodes among workers
        episodes_per_worker = max(1, training_config.num_episodes // self.num_processes)
        remaining_episodes = training_config.num_episodes % self.num_processes
        
        # Prepare worker arguments
        worker_args = []
        for worker_id in range(self.num_processes):
            worker_episodes = episodes_per_worker + (1 if worker_id < remaining_episodes else 0)
            worker_args.append((config, training_config, worker_id, worker_episodes))
        
        print(f"Starting parallel training with {self.num_processes} workers")
        print(f"Episodes per worker: {[args[3] for args in worker_args]}")
        
        # Run training in parallel
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            results = list(executor.map(self._train_worker, worker_args))
        
        total_time = time.time() - start_time
        
        # Combine results
        combined_stats = {
            'total_parallel_time': total_time,
            'speedup_factor': (training_config.num_episodes * 0.12) / total_time,  # Estimated vs actual
            'workers_used': self.num_processes,
            'worker_results': {}
        }
        
        for stats, worker_name in results:
            combined_stats['worker_results'][worker_name] = stats
        
        # Calculate aggregate statistics
        all_rewards = []
        total_episodes = 0
        for stats, _ in results:
            if 'avg_reward' in stats:
                all_rewards.append(stats['avg_reward'])
            if 'episodes_completed' in stats:
                total_episodes += stats['episodes_completed']
        
        if all_rewards:
            combined_stats['avg_reward_across_workers'] = np.mean(all_rewards)
            combined_stats['total_episodes_completed'] = total_episodes
        
        return combined_stats


class VectorizedEnvironment:
    """Vectorized environment operations for better performance."""
    
    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.envs = [Environment() for _ in range(num_envs)]
    
    def vectorized_step(self, all_actions: List[List]) -> Tuple[List, List, List, List]:
        """Execute steps in multiple environments simultaneously."""
        # Use threading for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            futures = [
                executor.submit(env.step, actions) 
                for env, actions in zip(self.envs, all_actions)
            ]
            results = [future.result() for future in futures]
        
        # Separate results
        observations = [r[0] for r in results]
        rewards = [r[1] for r in results] 
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        
        return observations, rewards, dones, infos


def benchmark_optimizations():
    """Benchmark different optimization strategies."""
    
    print("=== Benchmarking Optimization Strategies ===")
    
    # Create test configuration
    env = Environment()
    config = FieldDistribution()
    
    # Add test agents (smaller for benchmarking)
    for i in range(4):
        team = Team.BLUE if i < 2 else Team.WHITE
        role = 'midfielder'
        pos = np.array([(-15 if i < 2 else 15), (i % 2 - 0.5) * 10])
        config.add(Agent(env, team, role=role, pos=pos))
    
    # Benchmark configurations
    benchmarks = {
        'baseline': {'episodes': 10, 'parallel': False},
        'parallel_2': {'episodes': 10, 'parallel': True, 'processes': 2},
        'parallel_4': {'episodes': 10, 'parallel': True, 'processes': 4},
    }
    
    results = {}
    
    for name, bench_config in benchmarks.items():
        print(f"\nBenchmarking: {name}")
        
        training_config = TrainingConfig(
            num_episodes=bench_config['episodes'],
            checkpoint_frequency=bench_config['episodes'],
            stats_frequency=bench_config['episodes'],
            max_episode_steps=50
        )
        
        start_time = time.time()
        
        if bench_config['parallel']:
            trainer = ParallelTrainer(num_processes=bench_config.get('processes', 2))
            stats = trainer.parallel_train(config, training_config)
        else:
            stats = train(config, training_config)
            stats = {'baseline_time': time.time() - start_time}
        
        results[name] = {
            'total_time': time.time() - start_time,
            'episodes': bench_config['episodes'],
            'stats': stats
        }
    
    # Print benchmark results
    print(f"\n=== Benchmark Results ===")
    baseline_time = results['baseline']['total_time']
    
    for name, result in results.items():
        total_time = result['total_time']
        speedup = baseline_time / total_time if total_time > 0 else 0
        episodes = result['episodes']
        
        print(f"{name:12} | Time: {total_time:.3f}s | Episodes: {episodes:3d} | Speedup: {speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    benchmark_optimizations()