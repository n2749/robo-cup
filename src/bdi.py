from enum import Enum
from typing import Dict, Any, Optional, List
import numpy as np


class Beliefs:
    """
    Represents an agent's beliefs about the current state of the soccer
    environment. Enhanced with confidence tracking and temporal updates. These
    beliefs are used both for BDI reasoning and as state representation for
    Q-learning.
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
        
        # Teammate spacing beliefs
        self.teammates_too_close: int = 0
        self.teammates_nearby: List[float] = []  # Distances to nearby teammates
        self.closest_teammate_distance: Optional[float] = None
        
        # World model with confidence tracking
        # Ball tracking with confidence
        self.wm_ball: Optional[np.ndarray] = None  # μglobal position
        self.wm_ball_confidence: float = 1.0  # σsum_ball
        self.wm_ball_timestamp: float = 0.0  # τsum_ball
        self.saw_ball: bool = False  # sum_sawball
        
        # Opponent tracking with confidence  
        self.wm_opponents: List[dict] = []  # List of opponent observations
        self.wm_teammates: List[dict] = []  # List of teammate observations
        
        # Vision and localization
        self.wm_position: Optional[np.ndarray] = None  # Robot position
        self.wm_heading: Optional[float] = None  # Robot heading
        
        # Temporal parameters
        self.current_time: float = 0.0  # τcurrent
        self.confidence_threshold: float = 0.5  # σthreshold
        self.time_threshold: float = 5.0  # τthreshold
        self.small_error: float = 0.1  # SMALL_ERROR
        
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
        
        # Teammate spacing assessment
        self.teammates_too_close = belief_dict.get('teammates_too_close', 0)
        self.teammates_nearby = belief_dict.get('teammates_nearby', [])
        self.closest_teammate_distance = belief_dict.get('closest_teammate_distance', None)
    
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
            'opponent_threatening': self.opponent_threatening,
            'teammates_too_close': self.teammates_too_close,
            'teammates_nearby': self.teammates_nearby,
            'closest_teammate_distance': self.closest_teammate_distance,
            'wm_ball': self.wm_ball.tolist() if self.wm_ball is not None else None,
            'wm_ball_confidence': self.wm_ball_confidence,
            'current_time': self.current_time
        }
    
    def update_world_model(self, robot_position: np.ndarray, robot_angle: float, 
                          ball_pos: Optional[np.ndarray], op_pos: List[np.ndarray], 
                          current_time: float):
        """
        UPDATEWORLDMODEL procedure
        
        Args:
            robot_position: Robot position
            robot_angle: Robot heading angle
            ball_pos: Observed ball position (None if not seen)
            op_pos: List of observed opponent positions
            current_time: Current simulation time
        """
        self.current_time = current_time
        
        # UPDATELOCALIZATION
        self._update_localization(robot_position, robot_angle)
        
        # UPDATEVISION
        self._update_vision(ball_pos, op_pos, current_time)
        
        # UPDATESHAREDINFORMATION
        self._update_shared_information(current_time)
        
        # UPDATETIME
        self._update_time(current_time)
    
    def _update_localization(self, robot_position: np.ndarray, robot_angle: float):
        """
        UPDATELOCALIZATION procedure - simply copies localization info.
        """
        self.wm_position = robot_position.copy()
        self.wm_heading = robot_angle
    
    def _update_vision(self, ball_pos: Optional[np.ndarray], op_pos: List[np.ndarray], 
                      current_time: float):
        """
        UPDATEVISION procedure.
        Updates ball and opponent positions from vision module.
        """
        # Update ball position
        if ball_pos is not None:
            # Convert to global coordinates
            mu_global = self.wm_position + self._rotate(ball_pos, self.wm_heading)
            sigma_global = self.small_error  # SMALL_ERROR
            
            # MERGE with existing ball belief
            self.wm_ball = self._merge_position(self.wm_ball, mu_global, 
                                              self.wm_ball_confidence, sigma_global)
            self.wm_ball_confidence = sigma_global
            self.wm_ball_timestamp = current_time
            self.saw_ball = True
        
        # Update opponent positions
        for i, op_position in enumerate(op_pos):
            mu_global = self.wm_position + self._rotate(op_position, self.wm_heading)
            
            # Find closest existing opponent belief
            best_match_idx = -1
            min_distance = float('inf')
            
            for j, existing_opp in enumerate(self.wm_opponents):
                if 'position' in existing_opp:
                    dist = np.linalg.norm(existing_opp['position'] - mu_global)
                    if dist < min_distance:
                        min_distance = dist
                        best_match_idx = j
            
            op_threshold = 5.0  # OP_THRESHOLD
            
            if min_distance < op_threshold:
                # Update existing opponent
                sigma_global = self.small_error
                self.wm_opponents[best_match_idx] = {
                    'position': self._merge_position(
                        self.wm_opponents[best_match_idx]['position'],
                        mu_global, 
                        self.wm_opponents[best_match_idx].get('confidence', 1.0),
                        sigma_global
                    ),
                    'confidence': sigma_global,
                    'timestamp': current_time
                }
            else:
                # Find oldest opponent to replace or add new
                if len(self.wm_opponents) < 11:  # Max opponents in 11v11
                    self.wm_opponents.append({
                        'position': mu_global,
                        'confidence': self.small_error,
                        'timestamp': current_time
                    })
                else:
                    # Replace oldest
                    oldest_idx = min(range(len(self.wm_opponents)),
                                   key=lambda k: self.wm_opponents[k].get('timestamp', 0))
                    self.wm_opponents[oldest_idx] = {
                        'position': mu_global,
                        'confidence': self.small_error,
                        'timestamp': current_time
                    }
    
    def _update_shared_information(self, current_time: float):
        """
        Request ball location from shared world model if not seen recently.
        """
        # Check if ball hasn't been seen in a long time
        if current_time - self.wm_ball_timestamp > self.time_threshold:
            # Request from shared world model
            shared_ball = self.get_ball_location(current_time, robot_id=id(self))
            
            if shared_ball is not None:
                self.wm_ball = self._merge_position(self.wm_ball, shared_ball['position'],
                                                  self.wm_ball_confidence, 
                                                  shared_ball['confidence'])
                self.wm_ball_confidence = shared_ball['confidence']
                self.wm_ball_timestamp = current_time
    
    def _update_time(self, current_time: float):
        """
        Add error to standard deviations of objects not updated this time period.
        """
        # Update ball confidence if not seen this timestep
        if self.wm_ball_timestamp != current_time:
            self.wm_ball_confidence += self.small_error
        
        # Update opponent confidences
        for opponent in self.wm_opponents:
            if opponent.get('timestamp', 0) != current_time:
                opponent['confidence'] = opponent.get('confidence', 1.0) + self.small_error
    
    def get_ball_location(self, current_time: float, robot_id: int) -> Optional[dict]:
        """
        Get best ball location from shared world model.
        
        Args:
            current_time: Current time
            robot_id: ID of requesting robot
            
        Returns:
            Dictionary with ball position and confidence, or None
        """
        # This would typically query other agents or a shared world model
        # For now, return None (no shared information available)
        # In a full implementation, this would iterate through all agents
        # and find the one with the most confident ball observation
        return None
    
    def is_valid_observation(self, timestamp: float, confidence: float, 
                           current_time: float, saw_ball: bool) -> bool:
        """
        Args:
            timestamp: When observation was made
            confidence: Confidence of observation
            current_time: Current time
            saw_ball: Whether ball was actually seen
            
        Returns:
            True if observation is valid
        """
        # Time freshness check
        if current_time - timestamp > self.time_threshold:
            return False
        
        # Confidence check
        if confidence > self.confidence_threshold:
            return False
        
        # Data integrity check
        if not saw_ball:
            return False
        
        return True
    
    def _rotate(self, vector: np.ndarray, angle: float) -> np.ndarray:
        """
        ROTATE function for coordinate transformation.
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return rotation_matrix @ vector
    
    def _merge_position(self, pos1: Optional[np.ndarray], pos2: np.ndarray, 
                       conf1: float, conf2: float) -> np.ndarray:
        """
        MERGE function for combining position estimates with confidence weighting.
        """
        if pos1 is None:
            return pos2.copy()
        
        # Weight by inverse confidence (lower confidence = higher weight)
        w1 = 1.0 / (conf1 + 1e-6)
        w2 = 1.0 / (conf2 + 1e-6)
        total_weight = w1 + w2
        
        return (pos1 * w1 + pos2 * w2) / total_weight


class Desires:
    """
    Represents an agent's dynamic desires/goals in the BDI architecture.
    Desires adapt to game state, role, and tactical situations in real-time.
    These influence action selection and can be used to bias Q-learning rewards.
    """
    
    def __init__(self, role: str = "midfielder"):
        # Base desires (role-dependent defaults)
        self.role = role

        # PASS = 1
        # TACKLE = 2
        # SHOOT = 3
        # BLOCK = 4
        # MOVE = 5
        # STAY = 6
        
        # Current dynamic desires (updated each cycle)
        self.score_goal = 0.5
        self.move_towards_ball = 0.5
        self.defend_goal = 0.5
        self.steal_ball = 0.5
        self.block_opponent = 0.5
        self.maintain_position = 0.5
        self.support_teammate = 0.5
        self.preserve_energy = 0.3
        self.take_risks = 0.5
        self.disperse_from_teammates = 0.6  # Desire to maintain spacing
        
        # Base role desires (used as baseline)
        self.base_desires = self._get_base_desires_for_role(role)
        
        # Game state modifiers
        self.desperation_factor = 0.0  # Increases when losing
        self.confidence_factor = 1.0   # Increases when winning
        self.fatigue_factor = 0.0      # Increases over time
        
        # Initialize with base desires
        self._apply_base_desires()
    
    def _get_base_desires_for_role(self, role: str) -> dict:
        """Get base desire values for a specific role."""
        if role == 'attacker':
            return {
                'score_goal': 1.0,
                'move_towards_ball': 0.9,
                'defend_goal': 0.3,
                'take_risks': 0.7,
                'steal_ball': 0.4,
                'block_opponent': 0.3,
                'maintain_position': 0.4,
                'support_teammate': 0.6,
                'disperse_from_teammates': 0.5
            }
        elif role == 'defender':
            return {
                'defend_goal': 1.0,
                'steal_ball': 0.9,
                'block_opponent': 0.8,
                'score_goal': 0.2,
                'take_risks': 0.2,
                'move_towards_ball': 0.5,
                'maintain_position': 0.8,
                'support_teammate': 0.7,
                'disperse_from_teammates': 0.8
            }
        elif role == 'goalkeeper':
            return {
                'defend_goal': 1.0,
                'block_opponent': 1.0,
                'steal_ball': 0.6,
                'score_goal': 0.0,
                'take_risks': 0.1,
                'move_towards_ball': 0.3,
                'maintain_position': 0.9,
                'support_teammate': 0.5,
                'disperse_from_teammates': 0.3
            }
        else:  # midfielder
            return {
                'support_teammate': 0.8,
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
            self.support_teammate *= 1.2
            self.steal_ball *= 0.7
    
    def _apply_positional_modifications(self, beliefs: Beliefs):
        """Apply modifications based on field position."""
        
        # In attacking third - more attacking desires
        if beliefs.in_attacking_third:
            self.score_goal *= 1.4
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
        if beliefs.distance_to_ball and beliefs.distance_to_ball >= 4:
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
            'score_goal', 'move_towards_ball',
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
