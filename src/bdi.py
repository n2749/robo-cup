from enum import Enum
from typing import Dict, Any, Optional


class Beliefs:
    """
    Represents an agent's beliefs about the current state of the soccer environment.
    These beliefs are used both for BDI reasoning and as state representation for Q-learning.
    """
    
    def __init__(self):
        # Spatial beliefs
        self.distance_to_goal: Optional[float] = None
        self.distance_to_home_goal: Optional[float] = None
        self.distance_to_ball: Optional[float] = None
        self.distance_to_opponent: Optional[float] = None
        
        # Tactical beliefs
        self.teammate_open: Optional[bool] = None
        self.goal_open: Optional[bool] = None
        
        # Additional beliefs for better decision making
        self.has_ball_possession: bool = False
        self.team_has_possession: bool = False
        self.in_attacking_third: bool = False
        self.in_defensive_third: bool = False
        self.opponent_threatening: bool = False
        
    def update(self, belief_dict: Dict[str, Any]):
        """
        Update beliefs from environment observations.
        
        Args:
            belief_dict: Dictionary of belief updates from environment
        """
        # Update basic beliefs
        self.distance_to_goal = belief_dict.get('distance_to_goal')
        self.distance_to_home_goal = belief_dict.get('distance_to_home_goal')
        self.distance_to_ball = belief_dict.get('distance_to_ball')
        self.distance_to_opponent = belief_dict.get('distance_to_opponent')
        self.teammate_open = belief_dict.get('teammate_open')
        self.goal_open = belief_dict.get('goal_open')
        
        # Derive additional beliefs
        self.has_ball_possession = belief_dict.get('has_ball', False)
        self.team_has_possession = belief_dict.get('team_has_ball', False)
        
        # Tactical position assessment
        if self.distance_to_goal is not None:
            self.in_attacking_third = self.distance_to_goal < 30.0
            self.in_defensive_third = self.distance_to_home_goal < 30.0
        
        # Threat assessment
        if self.distance_to_opponent is not None:
            self.opponent_threatening = self.distance_to_opponent < 10.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert beliefs to dictionary for serialization or debugging"""
        return {
            'distance_to_goal': self.distance_to_goal,
            'distance_to_home_goal': self.distance_to_home_goal,
            'distance_to_ball': self.distance_to_ball,
            'distance_to_opponent': self.distance_to_opponent,
            'teammate_open': self.teammate_open,
            'goal_open': self.goal_open,
            'has_ball_possession': self.has_ball_possession,
            'team_has_possession': self.team_has_possession,
            'in_attacking_third': self.in_attacking_third,
            'in_defensive_third': self.in_defensive_third,
            'opponent_threatening': self.opponent_threatening
        }


class Desires:
    """
    Represents an agent's desires/goals in the BDI architecture.
    These influence action selection and can be used to bias Q-learning rewards.
    """
    
    def __init__(
        self,
        # Offensive desires
        score_goal: float = 1.0,
        keep_possession: float = 0.8,
        create_opportunities: float = 0.7,
        move_towards_ball: float = 0.6,
        
        # Defensive desires  
        defend_goal: float = 1.0,
        steal_ball: float = 0.8,
        block_opponent: float = 0.7,
        maintain_position: float = 0.5,
        
        # Tactical desires
        support_teammate: float = 0.6,
        preserve_energy: float = 0.3,
        take_risks: float = 0.4
    ):
        # Offensive desires
        self.score_goal = score_goal
        self.keep_possession = keep_possession
        self.create_opportunities = create_opportunities
        self.move_towards_ball = move_towards_ball
        
        # Defensive desires
        self.defend_goal = defend_goal
        self.steal_ball = steal_ball
        self.block_opponent = block_opponent
        self.maintain_position = maintain_position
        
        # Tactical desires
        self.support_teammate = support_teammate
        self.preserve_energy = preserve_energy
        self.take_risks = take_risks
        
    def update_based_on_role(self, role: str):
        """
        Adjust desires based on agent role (attacker, defender, etc.)
        
        Args:
            role: Agent role ('attacker', 'defender', 'midfielder')
        """
        if role == 'attacker':
            self.score_goal = 1.0
            self.move_towards_ball = 0.9
            self.create_opportunities = 0.8
            self.defend_goal = 0.3
            self.take_risks = 0.7
            
        elif role == 'defender':
            self.defend_goal = 1.0
            self.steal_ball = 0.9
            self.block_opponent = 0.8
            self.score_goal = 0.4
            self.take_risks = 0.2
            
        elif role == 'midfielder':
            self.keep_possession = 0.9
            self.support_teammate = 0.8
            self.create_opportunities = 0.7
            self.take_risks = 0.5
    
    def get_action_bias(self, action: 'Actions', beliefs: Beliefs) -> float:
        """
        Get desire-based bias for a specific action given current beliefs.
        This can be used to influence Q-learning action selection.
        
        Args:
            action: The action to evaluate
            beliefs: Current beliefs
            
        Returns:
            Bias value (higher = more desired)
        """
        bias = 0.0
        
        if action == Actions.SHOOT:
            bias += self.score_goal
            if beliefs.goal_open:
                bias += 0.3
                
        elif action == Actions.PASS:
            bias += self.keep_possession
            if beliefs.teammate_open:
                bias += 0.2
                
        elif action == Actions.MOVE:
            bias += self.move_towards_ball
            if beliefs.distance_to_ball and beliefs.distance_to_ball > 5.0:
                bias += 0.2
                
        elif action == Actions.TACKLE:
            bias += self.steal_ball
            if beliefs.opponent_threatening:
                bias += 0.3
                
        elif action == Actions.BLOCK:
            bias += self.defend_goal
            if beliefs.in_defensive_third:
                bias += 0.2
                
        elif action == Actions.STAY:
            bias += self.preserve_energy
            bias -= 0.1  # Generally discourage staying idle
        
        return bias


class Intentions:
    """
    Represents an agent's current intentions in the BDI architecture.
    These are the committed plans/actions the agent intends to execute.
    """
    
    def __init__(self):
        self.current_action: Optional['Actions'] = None
        self.action_sequence: list = []  # For multi-step plans
        self.commitment_strength: float = 0.5  # How committed to current plan
        self.plan_horizon: int = 1  # How many steps ahead to plan
        
    def set_intention(self, action: 'Actions', commitment: float = 0.7):
        """
        Set current intention with commitment strength.
        
        Args:
            action: The intended action
            commitment: How strongly committed to this action (0-1)
        """
        self.current_action = action
        self.commitment_strength = commitment
        
    def should_reconsider(self, new_beliefs: Beliefs) -> bool:
        """
        Determine if agent should reconsider current intentions based on new beliefs.
        
        Args:
            new_beliefs: Updated beliefs from environment
            
        Returns:
            True if agent should reconsider current plan
        """
        # Simple reconsideration logic - could be more sophisticated
        if self.commitment_strength < 0.3:
            return True
            
        # Reconsider if situation has changed dramatically
        if new_beliefs.opponent_threatening and self.current_action in [Actions.PASS, Actions.SHOOT]:
            return True
            
        if new_beliefs.goal_open and self.current_action != Actions.SHOOT:
            return True
            
        return False
        
    def decay_commitment(self, decay_rate: float = 0.9):
        """Reduce commitment over time"""
        self.commitment_strength *= decay_rate


# Actions available to soccer agents
class Actions(Enum):
    """Available actions for soccer agents in the BDI simulation"""
    PASS = 1
    TACKLE = 2
    SHOOT = 3
    BLOCK = 4
    MOVE = 5
    STAY = 6
    
    @classmethod
    def get_all_actions(cls):
        """Get list of all available actions"""
        return list(cls)
        
    @classmethod
    def get_offensive_actions(cls):
        """Get actions typically used in offensive situations"""
        return [cls.PASS, cls.SHOOT, cls.MOVE]
        
    @classmethod
    def get_defensive_actions(cls):
        """Get actions typically used in defensive situations"""
        return [cls.TACKLE, cls.BLOCK, cls.MOVE]


class BDIReasoningEngine:
    """
    Handles the BDI reasoning process: belief revision, goal selection, and planning.
    Integrates with Q-learning for action selection.
    """
    
    def __init__(self):
        self.belief_revision_enabled = True
        self.goal_reconsideration_threshold = 0.3
        
    def revise_beliefs(self, current_beliefs: Beliefs, observations: Dict[str, Any]) -> Beliefs:
        """
        Update beliefs based on new observations.
        
        Args:
            current_beliefs: Current belief state
            observations: New observations from environment
            
        Returns:
            Updated beliefs
        """
        if self.belief_revision_enabled:
            current_beliefs.update(observations)
        return current_beliefs
        
    def generate_options(self, beliefs: Beliefs, desires: Desires) -> list:
        """
        Generate possible actions (options) based on current beliefs and desires.
        
        Args:
            beliefs: Current beliefs
            desires: Agent's desires
            
        Returns:
            List of viable actions
        """
        options = []
        
        # Always consider basic movement
        options.append(Actions.MOVE)
        
        # Offensive options
        if beliefs.has_ball_possession:
            if beliefs.goal_open and desires.score_goal > 0.5:
                options.append(Actions.SHOOT)
            if beliefs.teammate_open and desires.support_teammate > 0.4:
                options.append(Actions.PASS)
                
        # Defensive options
        if not beliefs.team_has_possession:
            if beliefs.distance_to_opponent and beliefs.distance_to_opponent < 8.0:
                options.append(Actions.TACKLE)
            if beliefs.in_defensive_third:
                options.append(Actions.BLOCK)
                
        # Always have option to stay (though usually not preferred)
        options.append(Actions.STAY)
        
        return options
        
    def filter_options(self, options: list, beliefs: Beliefs, desires: Desires) -> list:
        """
        Filter options based on desires and current situation.
        
        Args:
            options: Available options
            beliefs: Current beliefs
            desires: Agent's desires
            
        Returns:
            Filtered list of options
        """
        filtered = []
        
        for action in options:
            # Calculate desire strength for this action
            desire_strength = desires.get_action_bias(action, beliefs)
            
            # Include action if desire is strong enough
            if desire_strength > 0.3:
                filtered.append(action)
                
        # Always include at least MOVE and STAY as fallbacks
        if Actions.MOVE not in filtered:
            filtered.append(Actions.MOVE)
        if len(filtered) == 1:  # If only MOVE, add STAY as alternative
            filtered.append(Actions.STAY)
            
        return filtered
