from enum import Enum
from abc import ABC
import numpy as np
from typing import List

from bdi import Beliefs, Desires, Intentions, Actions, BDIReasoningEngine
from qlearning import QLearningPolicy


class Team(Enum):
    WHITE = 1
    BLUE = 2


class Agent(ABC):
    """
    Base class for BDI soccer agents with Q-learning integration.
    Combines Belief-Desire-Intention architecture with reinforcement learning.
    """
    
    def __init__(self, env, team: Team, role: str = "midfielder", pos=np.zeros(2), base_pos=np.zeros(2)):
        # Environment and physical properties
        self.env = env
        self.env.register(self)
        self.team = team
        self.pos = pos.copy()
        self.base_pos = base_pos.copy()
        # Remember starting position so environment can restore it on reset
        self.initial_pos = self.pos.copy()
        self.vel = np.zeros(2)
        self.has_ball = False
        self.role = role
        
        # Positional zones for tactical play
        self._setup_zones()
        
        # BDI components
        self.beliefs = Beliefs()
        self.desires = Desires(role=role)  # Dynamic desires based on role
        self.intentions = Intentions()
        self.reasoning_engine = BDIReasoningEngine()
        
        # Q-learning integration
        self.q_policy = QLearningPolicy(
            actions=Actions.get_all_actions(),
            alpha=0.1,
            gamma=0.95,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=0.995
        )
        
        # Memory for learning
        self.previous_beliefs = None
        self.previous_action = None
        self.cumulative_reward = 0.0
        
        # Performance tracking
        self.episode_rewards = []
        self.actions_taken = []
    
    def _setup_zones(self):
        """Setup responsibility zones based on role and team"""
        field_width = self.env.width if self.env else 100.0
        field_height = self.env.height if self.env else 65.0
        
        # Team direction (attacking direction)
        if self.team.name == 'BLUE':
            # Blue attacks right (positive X)
            defensive_x = -field_width * 0.35
            midfield_x = 0.0
            attacking_x = field_width * 0.35
        else:
            # White attacks left (negative X)
            defensive_x = field_width * 0.35
            midfield_x = 0.0
            attacking_x = -field_width * 0.35
        
        # Define zones based on role
        if self.role == 'defender':
            self.zone_center = np.array([defensive_x, 0.0])
            self.zone_x_range = field_width * 0.25  # Can move 25% of field width
            self.zone_y_range = field_height * 0.4  # Can cover 40% of field height
        elif self.role == 'attacker':
            self.zone_center = np.array([attacking_x, 0.0])
            self.zone_x_range = field_width * 0.25
            self.zone_y_range = field_height * 0.4
        elif self.role == 'midfielder':
            self.zone_center = np.array([midfield_x, 0.0])
            self.zone_x_range = field_width * 0.3
            self.zone_y_range = field_height * 0.5
        else:  # goalkeeper
            goal_x = -field_width / 2 if self.team.name == 'BLUE' else field_width / 2
            self.zone_center = np.array([goal_x, 0.0])
            self.zone_x_range = field_width * 0.05
            self.zone_y_range = field_height * 0.3
    
    def is_in_zone(self) -> bool:
        """Check if agent is in their responsibility zone"""
        if not hasattr(self, 'zone_center'):
            return True
        
        dx = abs(self.pos[0] - self.zone_center[0])
        dy = abs(self.pos[1] - self.zone_center[1])
        
        return dx <= self.zone_x_range and dy <= self.zone_y_range
    
    def distance_from_zone(self) -> float:
        """Calculate how far agent is from their zone center"""
        if not hasattr(self, 'zone_center'):
            return 0.0
        
        return float(np.linalg.norm(self.pos - self.zone_center))
    
    def get_zone_return_direction(self) -> np.ndarray:
        """Get direction vector to return to zone"""
        if not hasattr(self, 'zone_center'):
            return np.zeros(2)
        
        direction = self.zone_center - self.pos
        if np.linalg.norm(direction) > 0:
            return direction / np.linalg.norm(direction)
        return np.zeros(2)
        
    
    def perceive(self):
        """
        Perceive environment and update beliefs through BDI reasoning.
        """
        # Get raw observations from environment
        observations = self.env.get_beliefs(self)
        
        # Add agent-specific observations
        observations['has_ball'] = self.has_ball
        observations['team_has_ball'] = any(agent.has_ball for agent in self.env.agents if agent.team == self.team)
        
        # Simulate vision system - ball position (None if not "seen")
        ball_pos = None
        if np.linalg.norm(self.pos - self.env.ball_pos) < 20.0:  # Vision range
            # Convert to local coordinates for vision simulation
            ball_pos = self.env.ball_pos - self.pos
        
        # Simulate opponent positions in vision
        op_positions = []
        for agent in self.env.agents:
            if agent.team != self.team and np.linalg.norm(self.pos - agent.pos) < 25.0:
                # Local coordinates
                op_positions.append(agent.pos - self.pos)
        
        self.beliefs.update_world_model(
            robot_position=self.pos,
            robot_angle=0.0,  # Simplified - no heading in current system
            ball_pos=ball_pos,
            op_pos=op_positions,
            current_time=self.env.current_time
        )
        
        # Update beliefs through BDI reasoning engine
        self.beliefs = self.reasoning_engine.revise_beliefs(self.beliefs, observations)
    
    
    def deliberate(self) -> List[Actions]:
        """
        BDI deliberation: generate possible actions based on beliefs and desires.
        
        Returns:
            List of viable actions
        """
        # Generate options through BDI reasoning
        options = self.reasoning_engine.generate_options(self.beliefs, self.desires)
        
        # Filter options based on desires and situation
        filtered_options = self.reasoning_engine.filter_options(options, self.beliefs, self.desires)
        
        return filtered_options if filtered_options else [Actions.MOVE, Actions.STAY]
    

    def plan(self, viable_actions: List[Actions], greedy: bool = False) -> Actions:
        """
        Select action using Q-learning with BDI bias.
        
        Args:
            viable_actions: Actions deemed viable by BDI deliberation
            greedy: If True, always select best Q-value action
            
        Returns:
            Selected action
        """
        # Check if we should reconsider current intentions
        if self.intentions.current_action is not None:
            if not self.intentions.should_reconsider(self.beliefs):
                # Stick with current intention if commitment is strong
                if self.intentions.current_action in viable_actions:
                    return self.intentions.current_action
        
        # Use Q-learning to select action, but only from viable BDI options
        if len(viable_actions) == 1:
            selected_action = viable_actions[0]
        else:
            # Temporarily modify Q-learning to only consider viable actions
            original_actions = self.q_policy.actions
            self.q_policy.actions = viable_actions
            
            # Select action with Q-learning
            selected_action = self.q_policy.select_action(self.beliefs, greedy=greedy)
            
            # Apply desire-based bias (influence exploration)
            if not greedy and np.random.random() < 0.2:  # 20% chance to apply bias
                action_scores = {}
                for action in viable_actions:
                    q_value = self.q_policy.Q[self.q_policy._q_key(self.q_policy._state_key(self.beliefs), action)]
                    desire_bias = self.desires.get_action_bias(action, self.beliefs)
                    action_scores[action] = q_value + desire_bias * 0.5
                
                selected_action = max(action_scores, key=action_scores.get)
            
            # Restore original actions
            self.q_policy.actions = original_actions
        
        # Set intention
        commitment_strength = 0.8 if selected_action in [Actions.SHOOT, Actions.PASS] else 0.6
        self.intentions.set_intention(selected_action, commitment_strength)
        
        return selected_action
    
    
    def act(self) -> Actions:
        """
        Execute the BDI reasoning cycle and return selected action.
        
        Returns:
            Action to execute
        """
        # BDI reasoning cycle
        self.perceive()
        
        # Update desires dynamically based on current game state
        game_info = self._get_game_info()
        self.desires.update_desires_based_on_game_state(self.beliefs, game_info)
        
        viable_actions = self.deliberate()
        selected_action = self.plan(viable_actions)
        
        # Store for learning
        self.previous_beliefs = Beliefs()
        self.previous_beliefs.update(self.beliefs.to_dict())
        self.previous_action = selected_action
        
        # Track actions
        self.actions_taken.append(selected_action)
        
        # Decay intention commitment
        self.intentions.decay_commitment()
        
        return selected_action
    
    def _get_game_info(self) -> dict:
        """
        Get current game information for dynamic desire updates.
        
        Returns:
            Dictionary with game state information
        """
        if not self.env:
            return {'my_team_score': 0, 'opponent_score': 0, 'time_remaining': 1.0, 'ball_possession': None}
        
        # Determine team scores
        if self.team.name == 'BLUE':
            my_score = self.env.score['blue']
            opponent_score = self.env.score['white']
        else:
            my_score = self.env.score['white']
            opponent_score = self.env.score['blue']
        
        # Calculate time remaining (0.0 = end of game, 1.0 = start of game)
        time_remaining = max(0.0, 1.0 - (self.env.current_step / self.env.episode_length))
        
        # Determine ball possession
        ball_possession = None
        if self.env.ball_owner:
            if self.env.ball_owner.team == self.team:
                ball_possession = 'my_team'
            else:
                ball_possession = 'opponent'
        
        return {
            'my_team_score': my_score,
            'opponent_score': opponent_score,
            'time_remaining': time_remaining,
            'ball_possession': ball_possession,
            'teammates_too_close': self.beliefs.teammates_too_close,
            'teammates_nearby': self.beliefs.teammates_nearby,
            'closest_teammate_distance': self.beliefs.closest_teammate_distance
        }
    
    
    def learn(self, reward: float, done: bool = False):
        """
        Update Q-learning based on received reward.
        
        Args:
            reward: Reward received for previous action
            done: Whether episode is finished
        """
        if self.previous_beliefs is not None and self.previous_action is not None:
            # Update Q-learning policy
            self.q_policy.update(
                self.previous_beliefs, 
                self.previous_action, 
                reward, 
                self.beliefs, 
                done
            )
            
            # Track performance
            self.cumulative_reward += reward
            
            if done:
                self.episode_rewards.append(self.cumulative_reward)
                self.cumulative_reward = 0.0
                self.q_policy.end_episode()
    
    
    def reset_episode(self):
        """
        Reset agent state for new episode.
        """
        self.pos = self.base_pos.copy()
        self.previous_beliefs = None
        self.previous_action = None
        self.cumulative_reward = 0.0
        self.has_ball = False
        self.intentions = Intentions()
        self.actions_taken = []
    
    
    def get_stats(self) -> dict:
        """
        Get learning and performance statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.q_policy.get_stats()
        stats.update({
            'role': self.role,
            'team': self.team.name,
            'recent_rewards': self.episode_rewards[-10:] if self.episode_rewards else [],
            'avg_recent_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0,
            'total_episodes': len(self.episode_rewards),
            'position': self.pos.tolist()
        })
        return stats


class Defender(Agent):
    """
    Defender agent with defensive-focused desires and positional discipline.
    Stays in defensive zone and passes forward when ball reaches zone boundary.
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2), base_pos=np.zeros(2)):
        super().__init__(env, team, role="defender", pos=pos, base_pos=base_pos)
    
    def deliberate(self) -> List[Actions]:
        """
        Defender-specific deliberation with zone awareness.
        """
        options = super().deliberate()
        
        # Check if out of zone
        out_of_zone = not self.is_in_zone()
        distance_from_zone = self.distance_from_zone()
        
        # If far from zone and don't have ball, prioritize returning
        if out_of_zone and not self.has_ball:
            if distance_from_zone > 15.0:
                # Must return to zone
                return [Actions.MOVE, Actions.STAY]
        
        # If have ball and near attacking zone boundary, should pass
        if self.has_ball:
            # Check if close to zone boundary in attacking direction
            if self.team.name == 'BLUE':
                at_boundary = self.pos[0] > (self.zone_center[0] + self.zone_x_range * 0.7)
            else:
                at_boundary = self.pos[0] < (self.zone_center[0] - self.zone_x_range * 0.7)
            
            if at_boundary:
                # At zone boundary with ball - should pass forward
                teammates = [a for a in self.env.agents if a.team == self.team and a.role == 'attacker']
                if teammates:
                    return [Actions.PASS, Actions.MOVE]
        
        # Normal defensive deliberation
        if self.beliefs.in_defensive_third:
            defensive_actions = [Actions.BLOCK, Actions.TACKLE, Actions.MOVE]
            options = [action for action in options if action in defensive_actions] + \
                     [action for action in options if action not in defensive_actions]
        
        return options


class Attacker(Agent):
    """
    Attacker agent with offensive focus and zone discipline.
    Stays in attacking zone, ready to receive passes and score.
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2), base_pos=np.zeros(2)):
        super().__init__(env, team, role="attacker", pos=pos, base_pos=base_pos)
    
    def deliberate(self) -> List[Actions]:
        """
        Attacker-specific deliberation with zone awareness.
        """
        options = super().deliberate()
        
        # Check zone position
        out_of_zone = not self.is_in_zone()
        distance_from_zone = self.distance_from_zone()
        
        # If far from attacking zone, return to position
        if out_of_zone and not self.has_ball:
            if distance_from_zone > 20.0:
                # Too far from attacking zone - return
                return [Actions.MOVE, Actions.STAY]
            elif distance_from_zone > 10.0:
                # Moderately far - prefer positioning
                return [Actions.MOVE, Actions.STAY, Actions.PASS]
        
        # If in attacking zone, focus on scoring
        if self.is_in_zone():
            if self.has_ball:
                # Have ball in attacking zone - shoot or pass
                if self.beliefs.goal_open or (self.beliefs.distance_to_goal and self.beliefs.distance_to_goal < 15.0):
                    return [Actions.SHOOT, Actions.PASS, Actions.MOVE]
                else:
                    return [Actions.PASS, Actions.MOVE, Actions.SHOOT]
            else:
                # In position without ball - stay ready to receive
                if self.beliefs.distance_to_ball and self.beliefs.distance_to_ball < 15.0:
                    return [Actions.MOVE, Actions.TACKLE, Actions.STAY]
                else:
                    # Ball far away - maintain position
                    return [Actions.STAY, Actions.MOVE]
        
        # Normal offensive deliberation
        if self.beliefs.in_attacking_third or self.beliefs.has_ball_possession:
            offensive_actions = [Actions.SHOOT, Actions.PASS, Actions.MOVE]
            options = [action for action in options if action in offensive_actions] + \
                     [action for action in options if action not in offensive_actions]
        
        return options


class Midfielder(Agent):
    """
    Midfielder agent with balanced desires and tactical focus.
    Desires are now dynamically managed by the BDI system.
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2), base_pos=np.zeros(2)):
        super().__init__(env, team, role="midfielder", pos=pos, base_pos=base_pos)
        # Desires are dynamically set based on role and game state


class Goalkeeper(Agent):
    """
    Goalkeeper agent with specialized defensive behaviors.
    Stays near goal and protects it. Does not chase the ball.
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2), base_pos=np.zeros(2)):
        super().__init__(env, team, role="goalkeeper", pos=pos, base_pos=base_pos)
        
        # Store goal position for goalkeeper
        if team.name == 'BLUE':
            self.goal_position = np.array([-env.width / 2, 0.0])
            self.goal_line_x = -env.width / 2
        else:
            self.goal_position = np.array([env.width / 2, 0.0])
            self.goal_line_x = env.width / 2
        
        self.max_distance_from_goal = 8.0  # Stay within 8m of goal
    
    
    def deliberate(self) -> List[Actions]:
        """
        Goalkeeper-specific deliberation focused on goal protection.
        Always stays near goal, never chases ball far from goal.
        """
        # Calculate distance from goal
        distance_from_goal = np.linalg.norm(self.pos - self.goal_position)
        
        # If too far from goal, return to goal area
        if distance_from_goal > self.max_distance_from_goal:
            return [Actions.MOVE, Actions.STAY]
        
        # If ball is close to goal area, can try to intercept
        if self.beliefs.distance_to_ball and self.beliefs.distance_to_ball < 10.0:
            if self.beliefs.distance_to_home_goal and self.beliefs.distance_to_home_goal < 15.0:
                # Ball is near our goal - active defense
                return [Actions.BLOCK, Actions.TACKLE, Actions.MOVE]
        
        # Default: stay in position and block
        return [Actions.BLOCK, Actions.STAY]
    



class FieldDistribution():
    def __init__(self, agents=None):
        self.agents = agents if agents is not None else []


    def add(self, agent: Agent):
        self.agents.append(agent)

