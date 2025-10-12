#!/usr/bin/env python3
"""
GA visualization for 11v11.

Loads a GA checkpoint produced by train_11v11_ga.py (containing best_gene) and
assigns the evolved GeneticPolicy to a chosen team, then launches the existing
pygame visualizer to watch matches.

Examples:
  # Use most recent GA checkpoint, GA on BLUE
  python src/visualize_11v11_ga.py

  # Specify checkpoint and put GA policy on WHITE team
  python src/visualize_11v11_ga.py \
    --checkpoint checkpoints/11v11_ga/ga_gen0005_20251006_232110.pkl \
    --team WHITE

  # Choose formations
  python src/visualize_11v11_ga.py --blue-formation 4-3-3 --white-formation 4-4-2
"""

import os
import pickle
import argparse
from typing import List

import numpy as np

from team_config import create_11v11_field_distribution, get_available_formations
from ga_policy import GeneticPolicy
from bdi import Actions
from visualize import SoccerVisualizer


def list_ga_checkpoints(checkpoint_dir: str = "checkpoints/11v11_ga") -> List[str]:
    if not os.path.exists(checkpoint_dir):
        return []
    files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith('.pkl') and f.startswith('ga_gen')
    ]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


essential_keys = {"best_gene"}

def load_ga_gene(checkpoint_path: str) -> np.ndarray:
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    if not essential_keys.issubset(set(data.keys())):
        raise ValueError("File does not look like a GA checkpoint with 'best_gene'.")
    gene = np.asarray(data['best_gene'], dtype=np.float32)
    if gene.ndim != 1:
        gene = gene.reshape(-1)
    return gene


def assign_policy_to_team(config, team_name: str, policy: GeneticPolicy):
    for agent in config.agents:
        if agent.team.name == team_name:
            agent.q_policy = policy.clone()


def main():
    parser = argparse.ArgumentParser(description='Visualize GA-evolved 11v11 team using pygame')
    parser.add_argument('--checkpoint', '-c', type=str, help='Path to GA checkpoint (from train_11v11_ga.py)')
    parser.add_argument('--checkpoint-dir', '-d', type=str, default='checkpoints/11v11_ga', help='Directory of GA checkpoints')
    parser.add_argument('--team', '-t', type=str, default='BLUE', choices=['BLUE', 'WHITE'], help='Which team uses GA policy')
    parser.add_argument('--blue-formation', '-b', type=str, default='4-4-2', help='BLUE formation (4-4-2, 4-3-3, 3-5-2)')
    parser.add_argument('--white-formation', '-w', type=str, default='4-4-2', help='WHITE formation (4-4-2, 4-3-3, 3-5-2)')
    parser.add_argument('--list', '-l', action='store_true', help='List recent GA checkpoints and exit')

    args = parser.parse_args()

    # List checkpoints
    if args.list:
        files = list_ga_checkpoints(args.checkpoint_dir)
        if not files:
            print(f"No GA checkpoints found in {args.checkpoint_dir}")
            return
        print("Available GA checkpoints (newest first):")
        for p in files[:20]:
            print("  ", os.path.basename(p))
        return

    # Resolve checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        files = list_ga_checkpoints(args.checkpoint_dir)
        if not files:
            print(f"No GA checkpoints found in {args.checkpoint_dir}. Run train_11v11_ga.py first.")
            return
        checkpoint_path = files[0]
        print(f"Using most recent GA checkpoint: {os.path.basename(checkpoint_path)}")

    if not os.path.exists(checkpoint_path):
        print(f"GA checkpoint not found: {checkpoint_path}")
        return

    # Load evolved gene and build policy
    try:
        gene = load_ga_gene(checkpoint_path)
    except Exception as e:
        print(f"Failed to load GA checkpoint: {e}")
        return

    policy = GeneticPolicy.from_gene(gene, eps_start=0.0, eps_end=0.0)

    # Build 11v11 teams
    formations = get_available_formations()
    if args.blue_formation not in formations or args.white_formation not in formations:
        print('Unknown formation. Available:', ', '.join(formations.keys()))
        return

    config = create_11v11_field_distribution(formations[args.blue_formation], formations[args.white_formation])

    # Assign GA policy to selected team
    assign_policy_to_team(config, args.team, policy)

    # Optionally: assign a neutral baseline GA to the other team (or keep their default q_policy)
    # Here we keep defaults for contrast.

    # Launch pygame visualizer using current env and agents
    visualizer = SoccerVisualizer(width=1400, height=900)
    visualizer.env = config.agents[0].env if config.agents else None
    visualizer.agents = list(config.agents)

    print("Starting GA visualization... (press Q to quit)")
    try:
        visualizer.run_continuous_visualization()
    except KeyboardInterrupt:
        print("\nVisualization interrupted.")
    finally:
        visualizer.close()


if __name__ == '__main__':
    main()
