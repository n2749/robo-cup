#!/usr/bin/env python3
"""
Genetic Algorithm training for 11v11 teams.

This script evolves the BLUE team policy (shared across all BLUE agents) using a
GeneticPolicy (chromosome-based linear policy), playing against a fixed baseline
WHITE team GeneticPolicy. It does not modify existing Q-learning code.

Usage examples:
- Quick small run
  python src/train_11v11_ga.py --generations 10 --population 12 --episodes 2

- Specify formations and mutation rate, checkpoint every gen
  python src/train_11v11_ga.py \
          --blue-formation 4-4-2 \
          --white-formation 4-3-3 \
          --generations 50 \
          --population 20 \
          --episodes 3 \
          --mutation-rate 0.1 \
          --checkpoint-dir checkpoints/11v11_ga
"""

import os
import time
import pickle
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np

from team_config import create_11v11_field_distribution, get_available_formations
from bdi import Actions
from ga_policy import GeneticPolicy, GAOps


def assign_team_policy(config, team_name: str, policy_template: GeneticPolicy):
    """Assign per-agent clones of a GeneticPolicy to a specific team in config."""
    for agent in config.agents:
        if agent.team.name == team_name:
            agent.q_policy = policy_template.clone()


def run_episode(config, max_steps: int = 600) -> Tuple[int, int]:
    """Simulate a single episode and return (blue_goals, white_goals)."""
    # Environment is shared through agents; get from first agent
    env = config.agents[0].env

    env.reset()
    for agent in config.agents:
        agent.reset_episode()

    done = False
    for _ in range(max_steps):
        # Select actions via agents (BDI + GA policy)
        actions = [agent.act() for agent in config.agents]
        # Step environment
        _, rewards, done, _ = env.step(actions)
        # Notify agents of rewards to keep internal tracking consistent
        for agent, r in zip(config.agents, rewards):
            agent.learn(float(r), done)
        if done:
            break

    return env.score['blue'], env.score['white']


def evaluate_individual(config, blue_gene: np.ndarray, white_policy: GeneticPolicy,
                         episodes: int, max_steps: int) -> float:
    """Evaluate a candidate blue gene and return average goal differential (BLUE - WHITE)."""
    stats = evaluate_individual_stats(config, blue_gene, white_policy, episodes, max_steps)
    return float(stats['avg_diff'])


def evaluate_individual_stats(config, blue_gene: np.ndarray, white_policy: GeneticPolicy,
                              episodes: int, max_steps: int) -> Dict[str, float]:
    """Evaluate a candidate and return detailed stats: avg_diff, avg_blue_goals, avg_white_goals, win_rate."""
    # Assign policies
    blue_policy = GeneticPolicy.from_gene(blue_gene, eps_start=0.0, eps_end=0.0)
    assign_team_policy(config, 'BLUE', blue_policy)
    assign_team_policy(config, 'WHITE', white_policy)

    diffs: List[float] = []
    blues: List[float] = []
    whites: List[float] = []
    wins = 0
    for _ in range(episodes):
        blue, white = run_episode(config, max_steps=max_steps)
        diffs.append(float(blue - white))
        blues.append(float(blue))
        whites.append(float(white))
        if blue > white:
            wins += 1
    avg_diff = float(np.mean(diffs)) if diffs else 0.0
    avg_blue = float(np.mean(blues)) if blues else 0.0
    avg_white = float(np.mean(whites)) if whites else 0.0
    win_rate = float(wins / episodes) if episodes > 0 else 0.0
    return {
        'avg_diff': avg_diff,
        'avg_blue_goals': avg_blue,
        'avg_white_goals': avg_white,
        'win_rate': win_rate,
    }


def save_checkpoint(out_dir: str, gen_idx: int, best_gene: np.ndarray, best_fitness: float,
                    history: List[float], args: argparse.Namespace, best_stats: Optional[Dict[str, float]] = None):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'ga_gen{gen_idx:04d}_{timestamp}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({
            'generation': gen_idx,
            'best_gene': best_gene,
            'best_fitness': best_fitness,
            'history': history,
'args': vars(args),
            'best_stats': best_stats or {},
        }, f)
    print(f"Saved GA checkpoint: {os.path.basename(path)}")


def main():
    parser = argparse.ArgumentParser(description='Genetic Algorithm training for 11v11 (BLUE team)')
    parser.add_argument('--blue-formation', '-b', type=str, default='4-4-2', help='BLUE formation (4-4-2, 4-3-3, 3-5-2)')
    parser.add_argument('--white-formation', '-w', type=str, default='4-4-2', help='WHITE formation (4-4-2, 4-3-3, 3-5-2)')
    parser.add_argument('--generations', '-g', type=int, default=30, help='Number of GA generations')
    parser.add_argument('--population', '-p', type=int, default=16, help='Population size')
    parser.add_argument('--episodes', '-e', type=int, default=3, help='Evaluation episodes per individual')
    parser.add_argument('--max-steps', '-s', type=int, default=600, help='Max steps per episode (20 Hz -> 30s at 600)')
    parser.add_argument('--mutation-rate', '-m', type=float, default=0.08, help='Per-gene mutation probability')
    parser.add_argument('--mutation-sigma', type=float, default=0.15, help='Stddev of Gaussian mutation noise')
    parser.add_argument('--tournament-k', type=int, default=3, help='Tournament size for selection')
    parser.add_argument('--elite', type=int, default=2, help='Number of elites to carry over')
    parser.add_argument('--checkpoint-dir', '-c', type=str, default='checkpoints/11v11_ga', help='Directory to store GA checkpoints')

    args = parser.parse_args()

    # Validate formations
    formations = get_available_formations()
    if args.blue_formation not in formations or args.white_formation not in formations:
        print('Unknown formation. Available:', ', '.join(formations.keys()))
        return

    # Build 11v11 config with chosen formations
    print('Setting up 11v11 teams...')
    config = create_11v11_field_distribution(formations[args.blue_formation], formations[args.white_formation])

    # Fixed WHITE baseline policy (frozen weights)
    white_baseline = GeneticPolicy(actions=list(Actions), eps_start=0.0, eps_end=0.0)
    # Bias WHITE to be mildly defensive: tweak BLOCK/MOVE weights towards lower risk
    # (These are small nudges; evolution on BLUE must overcome this baseline.)
    idx_block = list(Actions).index(Actions.BLOCK)
    idx_move = list(Actions).index(Actions.MOVE)
    white_baseline.weights[idx_block, 0] += 0.3  # bias term
    white_baseline.weights[idx_move, 0] += 0.1

    # GA setup
    num_features = 7
    pop = GAOps.init_population(args.population, num_features=num_features)
    fitness = np.zeros(args.population, dtype=np.float32)

    best_gene = pop[0].copy()
    best_fit = -1e9
    history: List[float] = []

    print('\nStarting GA training...')
    print(f"Generations: {args.generations}, Population: {args.population}, Episodes/ind: {args.episodes}")
    print(f"Mutation: rate={args.mutation_rate}, sigma={args.mutation_sigma}, Tournament k={args.tournament_k}, Elite={args.elite}")

    for gen in range(args.generations):
        start = time.time()

        # Evaluate population
        for i in range(args.population):
            fitness[i] = evaluate_individual(config, pop[i], white_baseline, args.episodes, args.max_steps)

        gen_best_idx = int(np.argmax(fitness))
        gen_best_fit = float(fitness[gen_best_idx])
        gen_avg_fit = float(np.mean(fitness))
        history.append(gen_best_fit)

        # Compute detailed stats for the best individual this generation
        best_stats = evaluate_individual_stats(
            config, pop[gen_best_idx], white_baseline, episodes=args.episodes, max_steps=args.max_steps
        )

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_gene = pop[gen_best_idx].copy()

        duration = time.time() - start
        print(
            f"Gen {gen+1:3d}/{args.generations} | best={gen_best_fit:+.3f} avg={gen_avg_fit:+.3f} "
            f"| blue={best_stats['avg_blue_goals']:.2f} white={best_stats['avg_white_goals']:.2f} "
            f"| win={best_stats['win_rate']:.2%} | time={duration:.1f}s"
        )

        # Elitism
        elite_count = max(0, min(args.elite, args.population))
        elite_idxs = np.argsort(-fitness)[:elite_count]
        elites = pop[elite_idxs].copy()

        # New population
        new_pop = []
        if elite_count > 0:
            new_pop.extend(list(elites))

        # Fill the rest with offspring
        while len(new_pop) < args.population:
            p1_idx = GAOps.tournament_select(pop, fitness, k=args.tournament_k)
            p2_idx = GAOps.tournament_select(pop, fitness, k=args.tournament_k)
            if p1_idx == p2_idx:
                p2_idx = (p2_idx + 1) % args.population
            child = GAOps.uniform_crossover(pop[p1_idx], pop[p2_idx], p=0.5)
            child = GAOps.mutate(child, sigma=args.mutation_sigma, rate=args.mutation_rate)
            new_pop.append(child)

        pop = np.stack(new_pop, axis=0).astype(np.float32)

        # Checkpoint after each generation (include best stats for convenience)
        save_checkpoint(args.checkpoint_dir, gen, best_gene, best_fit, history, args, best_stats=best_stats)

    print('\nGA training complete!')
    print(f"Best fitness (avg goal diff): {best_fit:+.3f}")
    print(f"Best checkpoint saved in: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
