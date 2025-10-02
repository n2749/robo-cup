import numpy as np
import random
from typing import List, Tuple, Optional
from bdi import Actions


class Environment:
    """
    Soccer field environment for BDI agents with Q-learning.
    Simulates a simplified 2D soccer field with ball physics and game rules.
    """
    
    def __init__(self, field_width: float = 100.0, field_height: float = 65.0):
        """
        Field Dimensions:
        - Soccer field is 2D rectangular, 100 x 65 meters
        - Center at (0,0), Y goes up, X goes right
        - Movements simulated stepwise for every 50 milliseconds (20 steps/sec)
        - Players and ball treated as circles
        """
        # Field dimensions (in meters)
        self.width = field_width
        self.height = field_height
        
        # Goal positions (center of each goal)
        # Center at (0,0), Y goes up, X goes right
        self.goal_left = np.array([0.0, field_height / 2])
        self.goal_right = np.array([field_width, field_height / 2])
        self.goal_width = 7.32  # FIFA standard goal width
        self.goal_height = 2.44  # For visualization reference
        
        # Game state
        self.ball_pos = np.array([field_width / 2, field_height / 2])  # Center
        self.ball_vel = np.zeros(2)
        self.ball_owner = None  # Agent that has possession
        
        # Registered agents
        self.agents = []
        
        self.timestep_ms = 50  # 50 milliseconds per step
        self.episode_length = 600  # 30 seconds at 20 steps/sec (600 steps)
        self.current_step = 0
        self.current_time = 0.0  # Time in seconds
        self.score = {'blue': 0, 'white': 0}
        self.episode_done = False
        
        # Collision tracking
        self.collisions_this_step = []
        
        # Football-specific areas
        self.penalty_area_width = 16.5  # Distance from goal line
        self.penalty_area_height = 40.3  # Width of penalty area
        self.center_circle_radius = 9.15
        self.corner_arc_radius = 1.0
        
        # Movement: P1 = P0 + V0; V1 = V0 + A0; A1 = FORCE * K1 - V0 * K2
        self.K1 = 0.1  # Force scaling factor
        self.K2 = 0.05  # Velocity damping factor
        self.max_force = 10.0  # Maximum force agents can apply
        
        # Ball physics parameters
        self.ball_friction_factor = 0.02  # FRICTIONFACTOR for ball
        self.kick_force_multiplier = 1.0  # KICKFORCE * K1
        self.max_ball_speed = 25.0  # Realistic ball speed (m/s)
        self.possession_distance = 1.5  # Distance to control ball
        
        # Collision parameters
        self.agent_radius = 1.5
        self.collision_velocity_multiplier = -0.1
        self.min_separation_force = 2.0
        
    def reset(self):
        """Reset environment for new episode"""
        # Reset ball to center
        self.ball_pos = np.array([self.width / 2, self.height / 2])
        self.ball_vel = np.zeros(2)
        self.ball_owner = None
        
        # Reset agents to starting positions
        for i, agent in enumerate(self.agents):
            if agent.team.name == 'BLUE':
                # Blue team starts on left side
                agent.pos = np.array([self.width * 0.25, self.height * (0.3 + 0.4 * (i % 2))])
            else:
                # White team starts on right side  
                agent.pos = np.array([self.width * 0.75, self.height * (0.3 + 0.4 * (i % 2))])
            
            agent.vel = np.zeros(2)
            agent.has_ball = False
            
        self.current_step = 0
        self.current_time = 0.0
        self.episode_done = False
        
        return self._get_observations()
    
    def register(self, agent):
        """Register an agent with the environment"""
        self.agents.append(agent)
    
    def step(self, actions: List[Actions]) -> Tuple[List, List[float], bool, dict]:
        """
        Execute one step of the simulation.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            observations, rewards, done, info
        """
        self.current_step += 1
        self.current_time += self.timestep_ms / 1000.0
        
        # Reset collision tracking
        self.collisions_this_step = []
        
        # Execute actions for each agent
        for agent, action in zip(self.agents, actions):
            self._execute_action(agent, action)
        
        # Update physics
        self._update_ball_physics()
        self._apply_repulsion_forces()  # Apply spacing forces
        self._handle_collisions()  # Handle agent-agent collisions
        self._update_possession()
        
        # Check for goals and boundaries
        goal_scored = self._check_goals()
        self._enforce_boundaries()
        
        # Calculate rewards
        rewards = self._calculate_rewards(actions, goal_scored)
        
        # Check if episode is done
        self.episode_done = (self.current_step >= self.episode_length) or goal_scored
        
        observations = self._get_observations()
        
        info = {
            'score': self.score.copy(),
            'goal_scored': goal_scored,
            'ball_pos': self.ball_pos.copy(),
            'step': self.current_step
        }
        
        return observations, rewards, self.episode_done, info
    
    def get_beliefs(self, agent) -> dict:
        """Get beliefs for a specific agent based on environment state"""
        # Calculate distances
        distance_to_ball = np.linalg.norm(agent.pos - self.ball_pos)
        
        # Distance to opponent's goal
        if agent.team.name == 'BLUE':
            distance_to_goal = np.linalg.norm(agent.pos - self.goal_right)
            distance_to_home_goal = np.linalg.norm(agent.pos - self.goal_left)
        else:
            distance_to_goal = np.linalg.norm(agent.pos - self.goal_left) 
            distance_to_home_goal = np.linalg.norm(agent.pos - self.goal_right)
        
        # Find nearest opponent
        opponents = [a for a in self.agents if a.team != agent.team]
        if opponents:
            distances_to_opponents = [np.linalg.norm(agent.pos - opp.pos) for opp in opponents]
            distance_to_opponent = min(distances_to_opponents)
        else:
            distance_to_opponent = float('inf')
        
        # Check if teammate is open (simplified)
        teammates = [a for a in self.agents if a.team == agent.team and a != agent]
        teammate_open = len(teammates) > 0 and any(
            np.linalg.norm(tm.pos - opp.pos) > 10.0 
            for tm in teammates for opp in opponents
        )
        
        # Check if goal is open (no opponents nearby)
        goal_pos = self.goal_right if agent.team.name == 'BLUE' else self.goal_left
        goal_open = all(
            np.linalg.norm(goal_pos - opp.pos) > 15.0 
            for opp in opponents
        )
        
        return {
            'distance_to_ball': distance_to_ball,
            'distance_to_goal': distance_to_goal,
            'distance_to_home_goal': distance_to_home_goal,
            'distance_to_opponent': distance_to_opponent,
            'teammate_open': teammate_open,
            'goal_open': goal_open
        }
    
    def _execute_action(self, agent, action: Actions):
        """Execute a single agent's action"""
        # A1 = FORCE * K1 - V0 * K2
        
        force = np.zeros(2)  # Default no force
        
        if action == Actions.MOVE:
            # Move towards ball with force
            direction = self.ball_pos - agent.pos
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                force = direction * self.max_force
        
        elif action == Actions.BLOCK:
            # Defensive positioning force towards own goal
            home_goal = np.array([0.0, self.height/2]) if agent.team.name == 'BLUE' else np.array([self.width, self.height/2])
            direction = home_goal - agent.pos
            distance_to_goal = np.linalg.norm(direction)
            
            if distance_to_goal > 10.0:  # Don't get too close to goal
                direction = direction / distance_to_goal
                force = direction * self.max_force * 0.7
        
        # A1 = FORCE * K1 - V0 * K2
        acceleration = force * self.K1 - agent.vel * self.K2
        
        # V1 = V0 + A0
        agent.vel += acceleration
        
        # P1 = P0 + V0  
        agent.pos += agent.vel
                
        elif action == Actions.PASS and agent.has_ball:
            # Pass to nearest teammate
            teammates = [a for a in self.agents if a.team == agent.team and a != agent]
            if teammates:
                nearest_teammate = min(teammates, key=lambda tm: np.linalg.norm(agent.pos - tm.pos))
                self._pass_ball(agent, nearest_teammate)
                
        elif action == Actions.SHOOT and agent.has_ball:
            # Shoot towards goal
            goal_pos = self.goal_right if agent.team.name == 'BLUE' else self.goal_left
            self._shoot_ball(agent, goal_pos)
            
        elif action == Actions.TACKLE:
            # Try to steal ball from nearby opponent
            opponents = [a for a in self.agents if a.team != agent.team and a.has_ball]
            for opp in opponents:
                if np.linalg.norm(agent.pos - opp.pos) < 5.0:
                    if random.random() < 0.3:  # 30% success rate
                        opp.has_ball = False
                        agent.has_ball = True
                        self.ball_owner = agent
                        break
                        
        elif action == Actions.BLOCK:
            # Defensive positioning (move towards own goal)
            home_goal = self.goal_left if agent.team.name == 'BLUE' else self.goal_right
            direction = home_goal - agent.pos
            distance_to_goal = np.linalg.norm(direction)
            
            if distance_to_goal > 10.0:  # Don't get too close to goal
                direction = direction / distance_to_goal
                # Slower movement for defensive positioning
                acceleration = direction * max_acceleration * 0.7
                agent.vel += acceleration
                
                # Limit velocity
                if np.linalg.norm(agent.vel) > max_velocity * 0.8:
                    agent.vel = (agent.vel / np.linalg.norm(agent.vel)) * max_velocity * 0.8
        
        elif action == Actions.STAY:
            # Apply strong damping to slow down
            agent.vel *= 0.5
        
        # Apply general velocity damping and update position
        agent.vel *= velocity_damping
        agent.pos += agent.vel
        
        # Very small velocities should be zeroed
        if np.linalg.norm(agent.vel) < 0.1:
            agent.vel = np.zeros(2)
    
    def _pass_ball(self, passer, receiver):
        """Execute a pass between agents"""
        direction = receiver.pos - passer.pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            self.ball_vel = (direction / distance) * min(10.0, distance * 0.5)
            passer.has_ball = False
            self.ball_owner = None
    
    def _shoot_ball(self, shooter, target_pos):
        """Execute a shot towards target"""
        direction = target_pos - shooter.pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # Add some randomness to shots
            noise = np.random.normal(0, 0.2, 2)
            direction = (direction / distance) + noise
            
            # If kicked, A1 = KICKFORCE * K1; V1 = 0
            kick_force = min(self.max_ball_speed, 8.0 + distance * 0.1)
            kick_acceleration = kick_force * self.kick_force_multiplier
            
            # Ball starts from rest when kicked (V1 = 0 initially, then gets acceleration)
            self.ball_vel = direction * kick_acceleration
            shooter.has_ball = False
            self.ball_owner = None
    
    def _update_ball_physics(self):
        # P1 = P0 + V0
        self.ball_pos += self.ball_vel
        
        # V1 = V0 + A0
        # A1 = -FRICTIONFACTOR * V0 (when not kicked)
        friction_acceleration = -self.ball_friction_factor * self.ball_vel
        self.ball_vel += friction_acceleration
        
        # Stop ball if moving very slowly
        if np.linalg.norm(self.ball_vel) < 0.1:
            self.ball_vel = np.zeros(2)
    
    def _update_possession(self):
        """Update which agent has ball possession"""
        if self.ball_owner is None or not self.ball_owner.has_ball:
            # Check if any agent is close enough to take possession
            for agent in self.agents:
                distance = np.linalg.norm(agent.pos - self.ball_pos)
                if distance < self.possession_distance:
                    agent.has_ball = True
                    self.ball_owner = agent
                    self.ball_vel = np.zeros(2)  # Ball stops when possessed
                    break
    
    def _check_goals(self) -> bool:
        """Check if a goal was scored"""
        # Left goal (Blue team defends this)
        if (self.ball_pos[0] <= 0 and 
            abs(self.ball_pos[1] - self.goal_left[1]) <= self.goal_width / 2):
            self.score['white'] += 1
            return True
        
        # Right goal (White team defends this)
        if (self.ball_pos[0] >= self.width and 
            abs(self.ball_pos[1] - self.goal_right[1]) <= self.goal_width / 2):
            self.score['blue'] += 1
            return True
        
        return False
    
    def _apply_repulsion_forces(self):
        """
        Apply repulsion forces between agents to maintain spacing.
        This prevents tight clustering by applying gentle pushes when agents get close.
        """
        repulsion_distance = self.agent_radius * 3.0  # Start repulsion at 3x collision radius
        max_repulsion_force = 0.5  # Maximum repulsion velocity
        
        for i, agent1 in enumerate(self.agents):
            repulsion_force = np.zeros(2)
            
            for j, agent2 in enumerate(self.agents):
                if i == j:
                    continue
                    
                # Calculate distance and direction
                distance_vec = agent1.pos - agent2.pos
                distance = np.linalg.norm(distance_vec)
                
                # Apply repulsion if agents are too close
                if 0 < distance < repulsion_distance:
                    if distance > 0:
                        # Normalize direction
                        direction = distance_vec / distance
                        
                        # Calculate repulsion strength (stronger when closer)
                        repulsion_strength = max_repulsion_force * (1.0 - distance / repulsion_distance)
                        
                        # Apply repulsion force
                        repulsion_force += direction * repulsion_strength
            
            # Apply the accumulated repulsion force
            if np.linalg.norm(repulsion_force) > 0:
                # Limit maximum repulsion
                if np.linalg.norm(repulsion_force) > max_repulsion_force:
                    repulsion_force = (repulsion_force / np.linalg.norm(repulsion_force)) * max_repulsion_force
                
                # Add to agent velocity
                agent1.vel += repulsion_force * 0.3  # Scale down the effect
    
    def _handle_collisions(self):
        """
        Handle collisions between agents.
        Uses elastic collision physics with damping.
        """
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                # Calculate distance between agents
                distance_vec = agent2.pos - agent1.pos
                distance = np.linalg.norm(distance_vec)
                
                # Check if collision occurs
                min_distance = self.agent_radius * 2
                if distance < min_distance and distance > 0:
                    # Normalize distance vector
                    normal = distance_vec / distance
                    
                    # Separate overlapping agents
                    overlap = min_distance - distance
                    separation_force = max(overlap * 0.6, self.min_separation_force)
                    separation = normal * separation_force
                    agent1.pos -= separation * 0.5
                    agent2.pos += separation * 0.5
                    
                    # Calculate relative velocity
                    relative_velocity = agent2.vel - agent1.vel
                    
                    # Calculate collision response (elastic collision with damping)
                    velocity_along_normal = np.dot(relative_velocity, normal)
                    
                    # Don't resolve if velocities are separating
                    if velocity_along_normal > 0:
                        continue
                    
                    # multiply velocities by -0.1 after collision
                    agent1.vel *= self.collision_velocity_multiplier
                    agent2.vel *= self.collision_velocity_multiplier
                    
                    # Add small random perturbation to prevent agents getting stuck
                    if distance < min_distance * 0.9:
                        perturbation = np.random.normal(0, 0.1, 2)
                        agent1.vel += perturbation
                        agent2.vel -= perturbation
                    
                    # Handle ball possession during collision
                    self._handle_collision_ball_transfer(agent1, agent2, normal)
                    
                    # Track collision for rewards/statistics
                    self.collisions_this_step.append({
                        'agents': (agent1, agent2),
                        'position': (agent1.pos + agent2.pos) / 2,
                        'force': np.linalg.norm(impulse)
                    })
    
    def _handle_collision_ball_transfer(self, agent1, agent2, collision_normal):
        """
        Handle ball possession transfer during agent collisions.
        
        Args:
            agent1, agent2: Colliding agents
            collision_normal: Normal vector of collision
        """
        # If one agent has the ball, there's a chance it gets knocked loose
        if agent1.has_ball:
            # Ball can be knocked loose or transferred
            if np.random.random() < 0.3:  # 30% chance ball gets knocked loose
                agent1.has_ball = False
                self.ball_owner = None
                # Ball flies in direction of collision
                self.ball_vel = collision_normal * np.random.uniform(2.0, 5.0)
            elif np.random.random() < 0.2:  # 20% chance ball transfers to other agent
                agent1.has_ball = False
                agent2.has_ball = True
                self.ball_owner = agent2
                self.ball_pos = agent2.pos.copy()
                
        elif agent2.has_ball:
            # Same logic for agent2
            if np.random.random() < 0.3:
                agent2.has_ball = False
                self.ball_owner = None
                self.ball_vel = -collision_normal * np.random.uniform(2.0, 5.0)
            elif np.random.random() < 0.2:
                agent2.has_ball = False
                agent1.has_ball = True
                self.ball_owner = agent1
                self.ball_pos = agent1.pos.copy()
    
    def _enforce_boundaries(self):
        """Keep ball and agents within field boundaries with collision response"""
        # Ball boundaries with bounce
        if self.ball_pos[0] <= 0 or self.ball_pos[0] >= self.width:
            self.ball_vel[0] *= -0.8  # Bounce with energy loss
            self.ball_pos[0] = np.clip(self.ball_pos[0], 0, self.width)
        
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.height:
            self.ball_vel[1] *= -0.8  # Bounce with energy loss
            self.ball_pos[1] = np.clip(self.ball_pos[1], 0, self.height)
        
        # Agent boundaries with collision response
        for agent in self.agents:
            # Handle boundary collisions for agents
            if agent.pos[0] <= self.agent_radius:
                agent.pos[0] = self.agent_radius
                if agent.vel[0] < 0:
                    agent.vel[0] *= -0.5  # Bounce back with damping
                    
            elif agent.pos[0] >= self.width - self.agent_radius:
                agent.pos[0] = self.width - self.agent_radius
                if agent.vel[0] > 0:
                    agent.vel[0] *= -0.5
            
            if agent.pos[1] <= self.agent_radius:
                agent.pos[1] = self.agent_radius
                if agent.vel[1] < 0:
                    agent.vel[1] *= -0.5
                    
            elif agent.pos[1] >= self.height - self.agent_radius:
                agent.pos[1] = self.height - self.agent_radius
                if agent.vel[1] > 0:
                    agent.vel[1] *= -0.5
    
    def _calculate_rewards(self, actions: List[Actions], goal_scored: bool) -> List[float]:
        """Calculate rewards for each agent including collision penalties"""
        rewards = []
        
        for agent, action in zip(self.agents, actions):
            reward = 0.0
            
            # Goal rewards
            if goal_scored:
                if agent.team.name == 'BLUE' and self.score['blue'] > 0:
                    reward += 100.0  # Goal scored
                elif agent.team.name == 'WHITE' and self.score['white'] > 0:
                    reward += 100.0  # Goal scored
                else:
                    reward -= 50.0  # Goal conceded
            
            # Ball possession reward
            if agent.has_ball:
                reward += 5.0
            
            # Distance to ball (encourage getting closer)
            distance_to_ball = np.linalg.norm(agent.pos - self.ball_pos)
            reward += max(0, 5.0 - distance_to_ball * 0.1)
            
            # Action-specific rewards
            if action == Actions.PASS and agent.has_ball:
                reward += 2.0
            elif action == Actions.SHOOT and agent.has_ball:
                reward += 3.0
            elif action == Actions.TACKLE:
                # Bonus if tackle was successful (simplified check)
                if agent.has_ball:
                    reward += 10.0
                # Small penalty for unnecessary tackling
                else:
                    reward -= 0.2
            
            # Small penalty for staying idle
            if action == Actions.STAY:
                reward -= 0.5
            
            # Collision penalties/rewards
            for collision in self.collisions_this_step:
                if agent in collision['agents']:
                    other_agent = collision['agents'][1] if collision['agents'][0] == agent else collision['agents'][0]
                    
                    # Penalty for collisions (promotes spacing)
                    reward -= 2.0
                    
                    # But reward if it's a defensive collision against opponent with ball
                    if other_agent.has_ball and other_agent.team != agent.team:
                        reward += 3.0  # Good defensive play
                    
                    # Higher penalty for colliding with teammate
                    elif other_agent.team == agent.team:
                        reward -= 4.0  # Strong penalty for poor teamwork
            
            # Reward for good positioning (spacing with teammates)
            teammates_too_close = 0
            teammates_well_spaced = 0
            
            for other_agent in self.agents:
                if other_agent != agent and other_agent.team == agent.team:
                    distance = np.linalg.norm(agent.pos - other_agent.pos)
                    
                    if distance < 5.0:  # Too close
                        teammates_too_close += 1
                    elif 5.0 <= distance <= 15.0:  # Well spaced
                        teammates_well_spaced += 1
            
            # Strong penalty for clustering with teammates
            if teammates_too_close > 0:
                reward -= 3.0 * teammates_too_close
            
            # Reward for maintaining good spacing
            if teammates_well_spaced > 0 and teammates_too_close == 0:
                reward += 1.0  # Bonus for good spacing
            
            # Velocity-based rewards (encourage controlled movement)
            speed = np.linalg.norm(agent.vel)
            if speed > 2.5:  # Too fast
                reward -= 0.2
            elif 0.5 < speed < 2.0:  # Good controlled speed
                reward += 0.1
            
            rewards.append(reward)
        
        return rewards
    
    def _get_observations(self) -> List[dict]:
        """Get observations for all agents"""
        return [self.get_beliefs(agent) for agent in self.agents]

