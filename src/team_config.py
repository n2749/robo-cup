#!/usr/bin/env python3
"""
11v11 Soccer Team Configuration Module
Defines standard soccer team formations and positions for full team simulation.
"""

import numpy as np
from typing import List, Tuple, Dict
from agents import Attacker, Defender, Midfielder, Goalkeeper, Team, FieldDistribution
from env import Environment


class SoccerFormation:
    """
    Base class for soccer formations with 11 players.
    """
    def __init__(self, name: str):
        self.name = name
        self.positions = []  # List of (role, x_ratio, y_ratio) tuples
        
    def get_positions(self, team: Team, field_width: float, field_height: float) -> List[Tuple[str, np.ndarray]]:
        """
        Get absolute positions for the formation based on team and field dimensions.
        
        Args:
            team: Team (BLUE attacks right, WHITE attacks left)
            field_width: Field width (100m)
            field_height: Field height (65m)
            
        Returns:
            List of (role, position) tuples
        """
        positions = []
        
        for role, x_ratio, y_ratio in self.positions:
            if team == Team.BLUE:
                # Blue team starts on left, attacks right
                x = -field_width/2 + x_ratio * field_width
            else:
                # White team starts on right, attacks left  
                x = field_width/2 - x_ratio * field_width
                
            y = (y_ratio - 0.5) * field_height
            positions.append((role, np.array([x, y])))
            
        return positions


class Formation442(SoccerFormation):
    """
    Classic 4-4-2 formation: 1 GK, 4 DEF, 4 MID, 2 ATT
    """
    def __init__(self):
        super().__init__("4-4-2")
        # Positions as (role, x_ratio, y_ratio) where x_ratio=0 is own goal, x_ratio=1 is opponent goal
        # y_ratio=0 is bottom, y_ratio=1 is top
        self.positions = [
            # Goalkeeper
            ("goalkeeper", 0.05, 0.5),
            
            # Defenders (4)
            ("defender", 0.20, 0.2),   # Right back
            ("defender", 0.20, 0.4),   # Right center back
            ("defender", 0.20, 0.6),   # Left center back  
            ("defender", 0.20, 0.8),   # Left back
            
            # Midfielders (4)
            ("midfielder", 0.45, 0.25),  # Right midfielder
            ("midfielder", 0.40, 0.42),  # Right center midfielder
            ("midfielder", 0.40, 0.58),  # Left center midfielder
            ("midfielder", 0.45, 0.75),  # Left midfielder
            
            # Attackers (2)
            ("attacker", 0.70, 0.4),   # Right forward
            ("attacker", 0.70, 0.6),   # Left forward
        ]


class Formation433(SoccerFormation):
    """
    Modern 4-3-3 formation: 1 GK, 4 DEF, 3 MID, 3 ATT
    """
    def __init__(self):
        super().__init__("4-3-3")
        self.positions = [
            # Goalkeeper
            ("goalkeeper", 0.05, 0.5),
            
            # Defenders (4)
            ("defender", 0.20, 0.15),  # Right back
            ("defender", 0.18, 0.38),  # Right center back
            ("defender", 0.18, 0.62),  # Left center back
            ("defender", 0.20, 0.85),  # Left back
            
            # Midfielders (3)
            ("midfielder", 0.50, 0.3),   # Right midfielder
            ("midfielder", 0.45, 0.5),   # Center midfielder
            ("midfielder", 0.50, 0.7),   # Left midfielder
            
            # Attackers (3)
            ("attacker", 0.75, 0.25),   # Right winger
            ("attacker", 0.80, 0.5),    # Center forward
            ("attacker", 0.75, 0.75),   # Left winger
        ]


class Formation352(SoccerFormation):
    """
    Attacking 3-5-2 formation: 1 GK, 3 DEF, 5 MID, 2 ATT
    """
    def __init__(self):
        super().__init__("3-5-2")
        self.positions = [
            # Goalkeeper
            ("goalkeeper", 0.05, 0.5),
            
            # Defenders (3)
            ("defender", 0.22, 0.3),   # Right center back
            ("defender", 0.20, 0.5),   # Center back
            ("defender", 0.22, 0.7),   # Left center back
            
            # Midfielders (5)
            ("midfielder", 0.35, 0.15),  # Right wing back
            ("midfielder", 0.45, 0.35),  # Right center midfielder
            ("midfielder", 0.40, 0.5),   # Center midfielder
            ("midfielder", 0.45, 0.65),  # Left center midfielder
            ("midfielder", 0.35, 0.85),  # Left wing back
            
            # Attackers (2)
            ("attacker", 0.70, 0.4),   # Right forward
            ("attacker", 0.70, 0.6),   # Left forward
        ]


def create_team(team: Team, formation: SoccerFormation, env: Environment) -> List:
    """
    Create a complete team with the specified formation.
    
    Args:
        team: Team (BLUE or WHITE)
        formation: Formation to use
        env: Environment instance
        
    Returns:
        List of agent instances
    """
    agents = []
    positions = formation.get_positions(team, env.width, env.height)
    
    for role, position in positions:
        if role == "goalkeeper":
            agent = Goalkeeper(env, team, pos=position)
        elif role == "defender":
            agent = Defender(env, team, pos=position)
        elif role == "midfielder":
            agent = Midfielder(env, team, pos=position)
        elif role == "attacker":
            agent = Attacker(env, team, pos=position)
        else:
            agent = Midfielder(env, team, pos=position)  # Default
            
        agents.append(agent)
    
    return agents


def create_11v11_field_distribution(blue_formation=None, white_formation=None) -> FieldDistribution:
    """
    Create a complete 11v11 field distribution with specified formations.
    
    Args:
        blue_formation: Formation for blue team (defaults to 4-4-2)
        white_formation: Formation for white team (defaults to 4-4-2)
        
    Returns:
        FieldDistribution with 22 agents
    """
    # Default formations
    if blue_formation is None:
        blue_formation = Formation442()
    if white_formation is None:
        white_formation = Formation442()
    
    # Create environment
    env = Environment()
    
    # Create teams
    blue_team = create_team(Team.BLUE, blue_formation, env)
    white_team = create_team(Team.WHITE, white_formation, env)
    
    # Create field distribution
    config = FieldDistribution()
    
    # Add all players
    for agent in blue_team:
        config.add(agent)
    for agent in white_team:
        config.add(agent)
    
    print(f"Created 11v11 match:")
    print(f"  Blue team ({blue_formation.name}): {len(blue_team)} players")
    print(f"  White team ({white_formation.name}): {len(white_team)} players")
    print(f"  Total agents: {len(config.agents)}")
    
    return config


def get_available_formations() -> Dict[str, SoccerFormation]:
    """Get dictionary of available formations."""
    return {
        "4-4-2": Formation442(),
        "4-3-3": Formation433(), 
        "3-5-2": Formation352()
    }


def print_formation_info():
    """Print information about available formations."""
    formations = get_available_formations()
    
    print("Available formations:")
    for name, formation in formations.items():
        print(f"\n{name}:")
        positions = formation.get_positions(Team.BLUE, 100, 65)
        role_counts = {}
        for role, pos in positions:
            role_counts[role] = role_counts.get(role, 0) + 1
        
        for role, count in role_counts.items():
            print(f"  {role}: {count}")


if __name__ == "__main__":
    # Test formations
    print("Testing soccer formations...")
    print_formation_info()
    
    # Create a test 11v11 setup
    config = create_11v11_field_distribution()
    print(f"\nTest setup complete with {len(config.agents)} total players")