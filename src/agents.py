from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List

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
    
    def __init__(self, env, team: Team, role: str = "midfielder", pos=np.zeros(2)):
        # Environment and physical properties
        self.env = env
        self.env.register(self)
        self.team = team
        self.pos = pos.copy() if pos is not None else np.zeros(2)
        self.vel = np.zeros(2)
        self.has_ball = False
        self.role = role
        
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
            'ball_possession': ball_possession
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
    Defender agent with defensive-focused desires and behaviors.
    Desires are now dynamically managed by the BDI system.
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2)):
        super().__init__(env, team, role="defender", pos=pos)
        # Desires are dynamically set based on role and game state
    
    
    def deliberate(self) -> List[Actions]:
        """
        Defender-specific deliberation with focus on defensive actions.
        """
        options = super().deliberate()
        
        # Prioritize defensive actions
        if self.beliefs.in_defensive_third:
            defensive_actions = [Actions.BLOCK, Actions.TACKLE, Actions.MOVE]
            options = [action for action in options if action in defensive_actions] + \
                     [action for action in options if action not in defensive_actions]
        
        return options


class Attacker(Agent):
    """
    Attacker agent with offensive-focused desires and behaviors.
    Desires are now dynamically managed by the BDI system.
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2)):
        super().__init__(env, team, role="attacker", pos=pos)
        # Desires are dynamically set based on role and game state
    
    
    def deliberate(self) -> List[Actions]:
        """
        Attacker-specific deliberation with focus on offensive actions.
        """
        options = super().deliberate()
        
        # Prioritize offensive actions
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
    
    def __init__(self, env, team: Team, pos=np.zeros(2)):
        super().__init__(env, team, role="midfielder", pos=pos)
        # Desires are dynamically set based on role and game state


class Goalkeeper(Agent):
    """
    Goalkeeper agent with specialized defensive behaviors.
    Desires are now dynamically managed by the BDI system.
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2)):
        super().__init__(env, team, role="goalkeeper", pos=pos)
        # Desires are dynamically set based on role and game state
    
    
    def deliberate(self) -> List[Actions]:
        """
        Goalkeeper-specific deliberation focused on goal protection.
        """
        # Goalkeepers primarily block and stay in position
        if self.beliefs.in_defensive_third:
            return [Actions.BLOCK, Actions.TACKLE, Actions.STAY]
        else:
            return [Actions.MOVE, Actions.STAY]


class FreeKickTaker(Agent):
    """
    Role: take-corner, Strategy: pass
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2)):
        super().__init__(env, team, role="free-kick-taker", pos=pos)
        
        # Free-kick-taker specific desires
        self.desires.keep_possession = 1.0
        self.desires.support_teammate = 0.9
        self.desires.create_opportunities = 0.8
        self.desires.take_risks = 0.3
        
    def deliberate(self) -> List[Actions]:
        """
        Free-kick-taker deliberation - focused on passing and corner kicks.
        """
        options = super().deliberate()
        
        # Prioritize passing for set pieces
        if self.beliefs.has_ball_possession:
            return [Actions.PASS, Actions.MOVE, Actions.STAY]
        else:
            return options
    
    def take_corner(self):
        """
        Execute corner kick - specialized set piece behavior.
        """
        # This would be called during corner kick situations
        # Find best teammate to pass to
        teammates = [a for a in self.env.agents if a.team == self.team and a != self]
        if teammates and self.has_ball:
            # Strategic corner kick pass
            best_target = max(teammates, key=lambda tm: tm.desires.score_goal)
            self.env._pass_ball(self, best_target)


class Player(Agent):
    """
    Role: position, move-ball, Strategies: go-zone, pass
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2)):
        super().__init__(env, team, role="player", pos=pos)
        
        # General player desires - balanced
        self.desires.keep_possession = 0.8
        self.desires.support_teammate = 0.7
        self.desires.move_towards_ball = 0.6
        self.desires.maintain_position = 0.5
        
    def deliberate(self) -> List[Actions]:
        """
        General player deliberation - balanced approach.
        """
        options = super().deliberate()
        
        # Add positional play
        if not self.beliefs.has_ball_possession:
            # Go to zone strategy
            return [Actions.MOVE, Actions.PASS, Actions.STAY]
        else:
            # Ball possession - pass or move
            return [Actions.PASS, Actions.MOVE, Actions.SHOOT]
    
    def go_zone(self, target_zone: np.ndarray):
        """
        Go-zone strategy - move to tactical position.
        """
        # Move towards assigned tactical zone
        direction = target_zone - self.pos
        if np.linalg.norm(direction) > 2.0:  # If not in position
            return Actions.MOVE
        else:
            return Actions.STAY


class FieldDistribution():
    def __init__(self, agents=None):
        self.agents = agents if agents is not None else []


    def add(self, agent: Agent):
        self.agents.append(agent)

