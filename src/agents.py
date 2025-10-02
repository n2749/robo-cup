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
        self.desires = Desires()
        self.desires.update_based_on_role(role)  # Configure desires based on role
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
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2)):
        super().__init__(env, team, role="defender", pos=pos)
        
        # Enhance defensive desires
        self.desires.defend_goal = 1.0
        self.desires.steal_ball = 0.9
        self.desires.block_opponent = 0.8
        self.desires.score_goal = 0.3
        self.desires.take_risks = 0.2
    
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
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2)):
        super().__init__(env, team, role="attacker", pos=pos)
        
        # Enhance offensive desires
        self.desires.score_goal = 1.0
        self.desires.move_towards_ball = 0.9
        self.desires.create_opportunities = 0.8
        self.desires.defend_goal = 0.4
        self.desires.take_risks = 0.7
    
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
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2)):
        super().__init__(env, team, role="midfielder", pos=pos)
        
        # Balanced desires
        self.desires.keep_possession = 0.9
        self.desires.support_teammate = 0.8
        self.desires.create_opportunities = 0.7
        self.desires.take_risks = 0.5


class Goalkeeper(Agent):
    """
    Goalkeeper agent with specialized defensive behaviors.
    """
    
    def __init__(self, env, team: Team, pos=np.zeros(2)):
        super().__init__(env, team, role="goalkeeper", pos=pos)
        
        # Goalkeeper-specific desires
        self.desires.defend_goal = 1.0
        self.desires.block_opponent = 1.0
        self.desires.steal_ball = 0.6
        self.desires.score_goal = 0.0  # Goalkeepers don't score
        self.desires.take_risks = 0.1
        self.desires.maintain_position = 0.9
    
    def deliberate(self) -> List[Actions]:
        """
        Goalkeeper-specific deliberation focused on goal protection.
        """
        # Goalkeepers primarily block and stay in position
        if self.beliefs.in_defensive_third:
            return [Actions.BLOCK, Actions.TACKLE, Actions.STAY]
        else:
            return [Actions.MOVE, Actions.STAY]

