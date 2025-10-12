from enum import Enum
from typing imporsupport_teammate': 0.8,
                'create_opportunities': 0.7,
                'take_risks': 0.5,
                'score_goal': 0.6,
                'defend_goal': 0.6,
                'move_towards_ball': 0.7,
                'steal_ball': 0.6,
                'block_opponent': 0.5,
                'maintain_position': 0.6,
                'disperse_from_teammates': 0.7
            }
    
    def _apply_base_desires(self):
        """Apply base desires for the agent's role."""
        for desire, value in self.base_desires.items():
            setattr(self, desire, value)
    
    def update_desires_based_on_game_state(self, beliefs: Beliefs, game_info: dict):
        """
        Dynamically update desires based on current game state.
        This is the core of dynamic BDI - desires change with context!
        
        Args:
            beliefs: Current agent beliefs
            game_info: Game state information (score, time, etc.)
        """
        # Start with base desires
        self._apply_base_desires()
        
        # Extract game state information
        my_score = game_info.get('my_team_score', 0)
        opponent_score = game_info.get('opponent_score', 0)
        game_time_remaining = game_info.get('time_remaining', 1.0)  # 0.0 to 1.0
        ball_possession = game_info.get('ball_possession', None)  # 'my_team', 'opponent', None
        
        # Calculate game situation factors
        score_difference = my_score - opponent_score
        is_losing = score_difference < 0
        is_winning = score_difference > 0
        is_tied = score_difference == 0
        time_pressure = 1.0 - game_time_remaining  # Higher when less time remaining
        
        # Update desperation and confidence factors
        if is_losing:
            self.desperation_factor = min(1.0, abs(score_difference) * 0.3 + time_pressure * 0.5)
            self.confidence_factor = max(0.5, 1.0 - self.desperation_factor)
        elif is_winning:
            self.confidence_factor = min(1.5, 1.0 + score_difference * 0.2)
            self.desperation_factor = max(0.0, time_pressure * 0.3 - 0.1)
        else:  # tied
            self.desperation_factor = time_pressure * 0.4
            self.confidence_factor = 1.0
        
        # Update fatigue factor (could be enhanced with actual agent energy)
        self.fatigue_factor = min(0.5, (1.0 - game_time_remaining) * 0.6)
        
        # Apply situational modifications
        self._apply_situational_modifications(beliefs, ball_possession, is_losing, is_winning, time_pressure)
        
        # Apply positional modifications based on field position
        self._apply_positional_modifications(beliefs)
        
        # Apply tactical modifications based on team state
        self._apply_tactical_modifications(beliefs, ball_possession)
        
        # Apply spacing modifications to prevent clustering
        self._apply_spacing_modifications(beliefs, game_info)
        
        # Ensure desires stay within reasonable bounds [0, 1.5]
        self._clamp_desires()
    
    def _apply_situational_modifications(self, beliefs: Beliefs, ball_possession: str, 
                                       is_losing: bool, is_winning: bool, time_pressure: float):
        """Apply modifications based on game situation."""
        
        # When losing - increase attacking desires
        if is_losing:
            self.score_goal *= (1.0 + self.desperation_factor * 0.5)
            self.take_risks *= (1.0 + self.desperation_factor * 0.4)
            self.move_towards_ball *= (1.0 + self.desperation_factor * 0.3)
            self.defend_goal *= (1.0 - self.desperation_factor * 0.2)  # Less defensive when desperate
            
        # When winning - more conservative
        elif is_winning:
            self.defend_goal *= self.confidence_factor
            self.take_risks *= (1.0 - (self.confidence_factor - 1.0) * 0.3)
            
        # Time pressure effects
        if time_pressure > 0.7:  # Last 30% of game
            if is_losing:
                # Desperate attacking in final minutes
                self.score_goal *= (1.0 + time_pressure * 0.8)
                self.take_risks *= (1.0 + time_pressure * 0.6)
            elif is_winning:
                # Defensive in final minutes when winning
                self.defend_goal *= (1.0 + time_pressure * 0.4)
                self.maintain_position *= (1.0 + time_pressure * 0.3)
                self.take_risks *= (1.0 - time_pressure * 0.4)
        
        # Ball possession effects
        if ball_possession == 'opponent':
            # Increase defensive desires when opponent has ball
            self.steal_ball *= 1.4
            self.block_opponent *= 1.3
            self.defend_goal *= 1.2
            self.score_goal *= 0.8
        elif ball_possession == 'my_team':
            # Increase attacking desires when we have ball
            self.score_goal *= 1.2
            self.create_opportunities *= 1.3
            self.support_teammate *= 1.2
            self.steal_ball *= 0.7
    
    def _apply_positional_modifications(self, beliefs: Beliefs):
        """Apply modifications based on field position."""
        
        # In attacking third - more attacking desires
        if beliefs.in_attacking_third:
            self.score_goal *= 1.4
            self.create_opportunities *= 1.3
            self.take_risks *= 1.2
            self.defend_goal *= 0.7
            
        # In defensive third - more defensive desires
        if beliefs.in_defensive_third:
            self.defend_goal *= 1.5
            self.block_opponent *= 1.3
            self.steal_ball *= 1.2
            self.score_goal *= 0.6
            self.take_risks *= 0.7
        
        # Close to ball - increase ball-related desires
        if beliefs.distance_to_ball and beliefs.distance_to_ball < 10.0:
            self.move_towards_ball *= 1.3
            if beliefs.distance_to_ball < 3.0:
                if beliefs.goal_open:
                    self.score_goal *= 1.6
        
        # Opponent threatening - emergency defense
        if beliefs.opponent_threatening:
            self.defend_goal *= 1.8
            self.block_opponent *= 1.6
            self.steal_ball *= 1.5
            self.score_goal *= 0.5
            self.take_risks *= 0.4
    
    def _apply_tactical_modifications(self, beliefs: Beliefs, ball_possession: str):
        """Apply modifications based on tactical situation."""
        
        # Teammate support
        if beliefs.teammate_open:
            self.support_teammate *= 1.4
            if ball_possession == 'my_team' and beliefs.has_ball_possession:
                self.create_opportunities *= 1.3  # Create plays when teammate is open
        
        # Goal opportunity
        if beliefs.goal_open:
            self.score_goal *= 1.8
            self.take_risks *= 1.5
            if beliefs.distance_to_goal and beliefs.distance_to_goal < 20.0:
                self.score_goal *= 2.0  # Strong desire to score when close and goal open
        
        # Energy management (fatigue effects)
        if self.fatigue_factor > 0.3:
            self.preserve_energy *= (1.0 + self.fatigue_factor)
            self.take_risks *= (1.0 - self.fatigue_factor * 0.5)
            self.move_towards_ball *= (1.0 - self.fatigue_factor * 0.3)
    
    def _apply_spacing_modifications(self, beliefs: Beliefs, game_info: dict):
        """Apply modifications to encourage proper teammate spacing and prevent clustering."""
        
        # Extract teammate proximity information if available
        teammates_nearby = game_info.get('teammates_nearby', [])
        teammates_too_close = game_info.get('teammates_too_close', 0)
        
        # Base dispersion desire already set from role
        base_disperse = self.disperse_from_teammates
        
        # Increase dispersion desire when teammates are too close
        if teammates_too_close > 0:
            # Strong desire to disperse when clustering occurs
            clustering_factor = min(2.0, 1.0 + teammates_too_close * 0.4)
            self.disperse_from_teammates *= clustering_factor
            
            # Also reduce some competing desires when clustering
            self.move_towards_ball *= 0.8  # Less focus on ball when need to spread
            self.support_teammate *= 0.9   # Less direct support, more spacing
        
        # If we're well spaced, maintain current positioning unless urgent
        elif len(teammates_nearby) > 0:
            # Good spacing - maintain but don't obsess over it
            self.disperse_from_teammates *= 1.0
            self.maintain_position *= 1.1  # Slight preference to maintain good position
        
        # Special cases based on game situation
        ball_possession = game_info.get('ball_possession', None)
        
        # When opponent has ball, allow closer spacing for defensive pressing
        if ball_possession == 'opponent':
            self.disperse_from_teammates *= 0.8  # Allow closer positioning for defense
            
        # When we have ball, ensure good spacing for passing options
        elif ball_possession == 'my_team':
            if not beliefs.has_ball_possession:  # I don't have ball, but team does
                self.disperse_from_teammates *= 1.3  # Create passing options
                self.support_teammate *= 1.2
        
        # In defensive third, allow closer spacing
        if beliefs.in_defensive_third:
            self.disperse_from_teammates *= 0.9
            
        # In attacking third, need good spacing for opportunities
        elif beliefs.in_attacking_third:
            self.disperse_from_teammates *= 1.2
    
    def _clamp_desires(self):
        """Ensure all desires stay within reasonable bounds."""
        desire_attributes = [
            'score_goal', 'create_opportunities', 'move_towards_ball',
            'defend_goal', 'steal_ball', 'block_opponent', 'maintain_position',
            'support_teammate', 'preserve_energy', 'take_risks',
            'disperse_from_teammates'
        ]
        
        for attr in desire_attributes:
            value = getattr(self, attr, 0.5)
            setattr(self, attr, max(0.0, min(1.5, value)))
    
    def get_current_desires_summary(self) -> dict:
        """Get current desire values for debugging/analysis."""
        return {
            'score_goal': self.score_goal,
            'create_opportunities': self.create_opportunities,
            'move_towards_ball': self.move_towards_ball,
            'defend_goal': self.defend_goal,
            'steal_ball': self.steal_ball,
            'block_opponent': self.block_opponent,
            'maintain_position': self.maintain_position,
            'support_teammate': self.support_teammate,
            'preserve_energy': self.preserve_energy,
            'take_risks': self.take_risks,
            'disperse_from_teammates': self.disperse_from_teammates,
            'desperation_factor': self.desperation_factor,
            'confidence_factor': self.confidence_factor,
            'fatigue_factor': self.fatigue_factor
        }
    
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
            if beliefs.teammate_open:
                bias += 0.2
                
        elif action == Actions.MOVE:
            bias += self.move_towards_ball
            if beliefs.distance_to_ball and beliefs.distance_to_ball > 5.0:
                bias += 0.2
            # Add bias for dispersing from teammates
            bias += self.disperse_from_teammates * 0.5
                
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
