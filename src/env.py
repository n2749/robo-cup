import numpy as np
import random
from typing import List, Tuple, Optional, TYPE_CHECKING
from bdi import Actions

if TYPE_CHECKING:
    from agents import Team


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
        # Goals are at the ends of the field
        self.goal_left = np.array([-field_width / 2, 0.0])  # Left goal at (-50, 0)
        self.goal_right = np.array([field_width / 2, 0.0])   # Right goal at (50, 0)
        self.goal_width = 7.32  # FIFA standard goal width
        self.goal_height = 2.44  # For visualization reference
        
        # Game state
        self.ball_pos = np.zeros(2)  # Center
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
        
        # Set piece state
        self.set_piece_type = None  # 'corner_kick', 'throw_in', 'goal_kick', None
        self.set_piece_team = None  # Team that gets the set piece
        self.set_piece_position = None  # Where the set piece happens
        self.set_piece_timer = 0  # Countdown before resuming play
        self.set_piece_duration = 100  # Steps to position before resuming (5 seconds at 20Hz, increased from 40)

        # Track last touch for better set-piece decisions
        self.last_touch_team = None
        
        # Movement: P1 = P0 + V0; V1 = V0 + A0; A1 = FORCE * K1 - V0 * K2
        self.K1 = 0.05  # Force scaling factor (slightly increased for better control)
        self.K2 = 0.12  # Velocity damping factor (more damping for stability)
        self.max_force = 6.0  # Maximum force agents can apply (slightly increased)
        
        # Ball physics parameters
        self.ball_friction_factor = 0.06  # FRICTIONFACTOR for ball (more friction)
        self.kick_force_multiplier = 0.7  # KICKFORCE * K1 (slightly increased)
        self.max_ball_speed = 18.0  # Realistic ball speed (m/s) (slightly increased)
        self.possession_distance = 2.0  # Distance to control ball (increased for easier possession)
        
        # Collision parameters
        self.agent_radius = 1.5
        self.collision_velocity_multiplier = -0.1
        self.min_separation_force = 2.0
        

    def reset(self):
        """Reset environment for new episode"""
        # Reset ball to center
        self.ball_pos = np.zeros(2)
        self.ball_vel = np.zeros(2)
        self.ball_owner = None
        
        # Reset agents to starting positions (field center at 0,0)
        for i, agent in enumerate(self.agents):
            # If agent has an initial_pos, use it; otherwise fall back to default layout
            if agent.base_pos is not None:
                agent.pos = agent.base_pos.copy()
            else:
                if agent.team.name == 'BLUE':
                    # Blue team starts on left side (negative X)
                    agent.pos = np.array([-self.width * 0.25, (i % 2 - 0.5) * self.height * 0.3])
                else:
                    # White team starts on right side (positive X)  
                    agent.pos = np.array([self.width * 0.25, (i % 2 - 0.5) * self.height * 0.3])
            
            agent.vel = np.zeros(2)
            agent.has_ball = False
            
        self.current_step = 0
        self.current_time = 0.0
        self.episode_done = False
        
        # Reset set piece state
        self.set_piece_type = None
        self.set_piece_team = None
        self.set_piece_position = None
        self.set_piece_timer = 0

        # Reset last touch
        self.last_touch_team = None
        
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
        
        goal_scored = False
        
        # Handle set pieces
        if self.set_piece_type is not None:
            self._handle_set_piece(actions)
        else:
            # Normal play
            # Execute actions for each agent
            for agent, action in zip(self.agents, actions):
                self._execute_action(agent, action)
            
            # Update physics
            self._update_ball_physics()
            self._apply_repulsion_forces()  # Apply spacing forces
            self._handle_collisions()  # Handle agent-agent collisions
            self._update_possession()
            
            # Check for goals FIRST (before out of bounds check)
            goal_scored = self._check_goals()
            
            # Only check out of bounds if no goal was scored
            if not goal_scored:
                self._check_out_of_bounds()
        
        # Calculate rewards
        rewards = self._calculate_rewards(actions, goal_scored)
        
        # Check if episode is done
        self.episode_done = (self.current_step >= self.episode_length) or goal_scored
        
        observations = self._get_observations()
        
        info = {
            'score': self.score.copy(),
            'goal_scored': goal_scored,
            'ball_pos': self.ball_pos.copy(),
            'step': self.current_step,
            'set_piece': self.set_piece_type
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
        
        # Calculate teammate spacing information
        teammates_nearby = []
        teammates_too_close = 0
        closest_teammate_distance = None
        
        if teammates:
            teammate_distances = [np.linalg.norm(agent.pos - tm.pos) for tm in teammates]
            closest_teammate_distance = min(teammate_distances)
            
            # Count teammates that are too close (within 5 meters)
            teammates_too_close = sum(1 for dist in teammate_distances if dist < 5.0)
            
            # List of distances to teammates within reasonable range (5-15 meters)
            teammates_nearby = [dist for dist in teammate_distances if 5.0 <= dist <= 15.0]
        
        return {
            'distance_to_ball': distance_to_ball,
            'distance_to_goal': distance_to_goal,
            'distance_to_home_goal': distance_to_home_goal,
            'distance_to_opponent': distance_to_opponent,
            'teammate_open': teammate_open,
            'goal_open': goal_open,
            'teammates_too_close': teammates_too_close,
            'teammates_nearby': teammates_nearby,
            'closest_teammate_distance': closest_teammate_distance
        }
    

    def _execute_action(self, agent, action: Actions):
        """Execute a single agent's action"""
        
        force = np.zeros(2)  # Default no force
        
        # Special handling for goalkeepers
        is_goalkeeper = hasattr(agent, 'role') and agent.role == 'goalkeeper'
        
        if action == Actions.MOVE:
            if is_goalkeeper:
                # Goalkeepers have special movement logic
                # They stay near goal and position based on ball
                if agent.team.name == 'BLUE':
                    goal_line_x = -self.width / 2
                else:
                    goal_line_x = self.width / 2
                
                # Calculate distance from goal
                goal_pos = np.array([goal_line_x, 0.0])
                distance_from_goal = np.linalg.norm(agent.pos - goal_pos)
                
                # If too far from goal, return to goal
                if distance_from_goal > 8.0:
                    direction = goal_pos - agent.pos
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        force = direction * self.max_force * 0.5
                # If ball is close to goal, position to intercept
                elif np.linalg.norm(self.ball_pos - goal_pos) < 20.0:
                    # Position on goal line based on ball Y position
                    target_y = np.clip(self.ball_pos[1], -self.height * 0.2, self.height * 0.2)
                    target_pos = np.array([goal_line_x + (3.0 if goal_line_x < 0 else -3.0), target_y])
                    direction = target_pos - agent.pos
                    if np.linalg.norm(direction) > 0.5:
                        direction = direction / np.linalg.norm(direction)
                        force = direction * self.max_force * 0.6
            else:
                # Normal field players movement with dispersion consideration
                ball_direction = self.ball_pos - agent.pos
                ball_distance = np.linalg.norm(ball_direction)
                
                # Calculate dispersion force from teammates
                dispersion_force = self._calculate_dispersion_force(agent)
                
                # Base force towards ball (if applicable)
                if ball_distance > 0:
                    ball_direction = ball_direction / ball_distance
                    ball_force = ball_direction * self.max_force
                else:
                    ball_force = np.zeros(2)
                
                # Combine ball attraction with dispersion
                # Strong dispersion desire overrides ball attraction when clustering
                if hasattr(agent, 'desires') and agent.desires.disperse_from_teammates > 1.0:
                    # Prioritize dispersion when strong clustering desire
                    force = dispersion_force * 0.7 + ball_force * 0.3
                else:
                    # Normal movement with some dispersion
                    force = ball_force * 0.8 + dispersion_force * 0.2
                
                # Ensure force doesn't exceed maximum
                if np.linalg.norm(force) > self.max_force:
                    force = (force / np.linalg.norm(force)) * self.max_force
                
        elif action == Actions.BLOCK:
            # Defensive positioning force towards own goal (consolidated handling)
            # Blue team defends left goal, White team defends right goal
            if agent.team.name == 'BLUE':
                home_goal = self.goal_left
            else:
                home_goal = self.goal_right
                
            direction = home_goal - agent.pos
            distance_to_goal = np.linalg.norm(direction)
            
            if distance_to_goal > 10.0:  # Don't get too close to goal
                direction = direction / distance_to_goal
                force = direction * self.max_force * 0.7  # Slower defensive movement
                
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
                        # Update last touch on successful tackle
                        self.last_touch_team = agent.team
                        break
        
        elif action == Actions.STAY:
            if is_goalkeeper:
                # Goalkeepers stay positioned on goal line
                if agent.team.name == 'BLUE':
                    goal_line_x = -self.width / 2
                else:
                    goal_line_x = self.width / 2
                
                # Small adjustment to optimal position based on ball
                target_y = np.clip(self.ball_pos[1] * 0.3, -self.height * 0.15, self.height * 0.15)
                target_pos = np.array([goal_line_x + (3.0 if goal_line_x < 0 else -3.0), target_y])
                direction = target_pos - agent.pos
                
                if np.linalg.norm(direction) > 0.5:
                    direction = direction / np.linalg.norm(direction)
                    force = direction * self.max_force * 0.3  # Very gentle adjustment
                else:
                    force = -agent.vel * self.K2 * 2.0  # Stop moving
            else:
                # Regular players apply strong damping to slow down
                force = -agent.vel * self.K2 * 2.0  # Extra damping for staying
        
        # A1 = FORCE * K1 - V0 * K2
        acceleration = force * self.K1 - agent.vel * self.K2
        
        # V1 = V0 + A0
        agent.vel += acceleration
        
        # P1 = P0 + V0
        agent.pos += agent.vel
        
        # Very small velocities should be zeroed
        if np.linalg.norm(agent.vel) < 0.1:
            agent.vel = np.zeros(2)
    
    
    def _pass_ball(self, passer, receiver):
        """Execute a pass between agents"""
        direction = receiver.pos - passer.pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # Smooth pass speed: scales with distance for realistic physics
            pass_speed = min(7.0, 3.0 + distance * 0.15)
            self.ball_vel = (direction / distance) * pass_speed
            passer.has_ball = False
            self.ball_owner = None
            # Track last touch team
            self.last_touch_team = passer.team
    
    
    def _shoot_ball(self, shooter, target_pos):
        """Execute a shot towards target"""
        direction = target_pos - shooter.pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # Add some randomness to shots (less randomness for more control)
            noise = np.random.normal(0, 0.1, 2)
            direction = (direction / distance) + noise
            direction = direction / np.linalg.norm(direction)  # Re-normalize
            
            # Shot power scales with distance: closer = harder shot
            kick_force = min(self.max_ball_speed, 5.0 + distance * 0.08)
            kick_acceleration = kick_force * self.kick_force_multiplier
            
            # Ball starts from rest when kicked (V1 = 0 initially, then gets acceleration)
            self.ball_vel = direction * kick_acceleration
            shooter.has_ball = False
            self.ball_owner = None
            # Track last touch team
            self.last_touch_team = shooter.team
    
    
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
                    # Update last touch team on possession
                    self.last_touch_team = agent.team
                    break
    
    def _check_goals(self) -> bool:
        """Check if a goal was scored"""
        # In soccer, a goal is scored when the ENTIRE ball crosses the goal line
        # We approximate the ball as having a small radius (0.11m = regulation soccer ball radius)
        ball_radius = 0.11  # Standard soccer ball radius in meters
        
        # Left goal at (-50, 0) - Blue team defends this
        # Ball must completely cross the goal line (ball center + radius beyond goal line)
        if (self.ball_pos[0] <= (-self.width / 2 - ball_radius) and 
            abs(self.ball_pos[1] - self.goal_left[1]) <= self.goal_width / 2):
            self.score['white'] += 1  # White team scores
            return True
        
        # Right goal at (50, 0) - White team defends this  
        # Ball must completely cross the goal line (ball center + radius beyond goal line)
        if (self.ball_pos[0] >= (self.width / 2 + ball_radius) and 
            abs(self.ball_pos[1] - self.goal_right[1]) <= self.goal_width / 2):
            self.score['blue'] += 1  # Blue team scores
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
    
    
    def _calculate_dispersion_force(self, agent) -> np.ndarray:
        """
        Calculate dispersion force to encourage agent to spread out from teammates.
        This is used specifically for MOVE actions when dispersion is desired.
        
        Args:
            agent: The agent to calculate dispersion force for
            
        Returns:
            Force vector pointing away from clustered teammates
        """
        teammates = [a for a in self.agents if a.team == agent.team and a != agent]
        if not teammates:
            return np.zeros(2)
        
        dispersion_force = np.zeros(2)
        influence_distance = 8.0  # Distance within which teammates influence dispersion
        
        for teammate in teammates:
            distance_vec = agent.pos - teammate.pos
            distance = np.linalg.norm(distance_vec)
            
            if 0 < distance < influence_distance:
                # Normalize direction (away from teammate)
                if distance > 0:
                    direction = distance_vec / distance
                    
                    # Stronger dispersion force when closer
                    dispersion_strength = (1.0 - distance / influence_distance) ** 2
                    dispersion_force += direction * dispersion_strength * self.max_force * 0.5
        
        # Also consider zone return if agent has a designated zone
        if hasattr(agent, 'is_in_zone') and hasattr(agent, 'get_zone_return_direction'):
            if not agent.is_in_zone():
                # Add force to return to zone when dispersing
                zone_direction = agent.get_zone_return_direction()
                dispersion_force += zone_direction * self.max_force * 0.3
        
        # Limit the dispersion force
        if np.linalg.norm(dispersion_force) > self.max_force:
            dispersion_force = (dispersion_force / np.linalg.norm(dispersion_force)) * self.max_force
        
        return dispersion_force
    
    
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
                    collision_force = separation_force  # Use separation force as collision magnitude
                    self.collisions_this_step.append({
                        'agents': (agent1, agent2),
                        'position': (agent1.pos + agent2.pos) / 2,
                        'force': collision_force
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
                # Last touch remains agent1's team
                self.last_touch_team = agent1.team
            elif np.random.random() < 0.2:  # 20% chance ball transfers to other agent
                agent1.has_ball = False
                agent2.has_ball = True
                self.ball_owner = agent2
                self.ball_pos = agent2.pos.copy()
                # Update last touch to new owner
                self.last_touch_team = agent2.team
                
        elif agent2.has_ball:
            # Same logic for agent2
            if np.random.random() < 0.3:
                agent2.has_ball = False
                self.ball_owner = None
                self.ball_vel = -collision_normal * np.random.uniform(2.0, 5.0)
                # Last touch remains agent2's team
                self.last_touch_team = agent2.team
            elif np.random.random() < 0.2:
                agent2.has_ball = False
                agent1.has_ball = True
                self.ball_owner = agent1
                self.ball_pos = agent1.pos.copy()
                # Update last touch to new owner
                self.last_touch_team = agent1.team
    
    
    def _check_out_of_bounds(self):
        """Check if ball went out of bounds and trigger appropriate set piece"""
        from agents import Team
        
        min_x, max_x = -self.width / 2, self.width / 2
        min_y, max_y = -self.height / 2, self.height / 2
        ball_radius = 0.11  # Same radius as in goal detection
        
        out_of_bounds = False
        
        # Check if ball crossed goal line (not in goal)
        # Ball is out when its center + radius crosses the line
        if self.ball_pos[0] <= (min_x - ball_radius) or self.ball_pos[0] >= (max_x + ball_radius):
            # Determine if it's a corner kick or goal kick
            if self.ball_pos[0] <= min_x + 0.5:
                # Ball crossed left goal line (outside goal area)
                if abs(self.ball_pos[1]) > self.goal_width / 2:
                    # Outside goal - corner kick or goal kick
                    last_touch = self._determine_last_touch()
                    if (last_touch is None) or (last_touch.name == 'BLUE'):
                        # Unknown last touch or Blue touched last: White gets corner
                        self._trigger_corner_kick(Team.WHITE, is_left_side=True, corner_top=(self.ball_pos[1] > 0))
                        out_of_bounds = True
                    else:
                        # White touched last: Blue gets goal kick
                        self._trigger_goal_kick(Team.BLUE, is_left_side=True)
                        out_of_bounds = True
            elif self.ball_pos[0] >= max_x - 0.5:
                # Ball crossed right goal line (outside goal area)
                if abs(self.ball_pos[1]) > self.goal_width / 2:
                    last_touch = self._determine_last_touch()
                    if (last_touch is None) or (last_touch.name == 'WHITE'):
                        # Unknown last touch or White touched last: Blue gets corner
                        self._trigger_corner_kick(Team.BLUE, is_left_side=False, corner_top=(self.ball_pos[1] > 0))
                        out_of_bounds = True
                    else:
                        # Blue touched last: White gets goal kick
                        self._trigger_goal_kick(Team.WHITE, is_left_side=False)
                        out_of_bounds = True
        
        # Check if ball crossed sideline (throw-in)
        if not out_of_bounds and (self.ball_pos[1] <= min_y + 0.5 or self.ball_pos[1] >= max_y - 0.5):
            last_touch = self._determine_last_touch()
            throw_in_x = np.clip(self.ball_pos[0], min_x, max_x)
            throw_in_y = max_y - 0.1 if self.ball_pos[1] >= max_y else min_y + 0.1
            
            # Determine which team gets throw-in
            if last_touch:
                # Opposite team gets throw-in
                throw_in_team = Team.WHITE if last_touch == Team.BLUE else Team.BLUE
            else:
                # If we can't determine last touch, randomly assign or use a default
                # For fairness, we'll randomly assign
                throw_in_team = Team.BLUE if np.random.random() < 0.5 else Team.WHITE
            
            self._trigger_throw_in(throw_in_team, np.array([throw_in_x, throw_in_y]))
            out_of_bounds = True
        
        # Keep ball in bounds if no set piece triggered
        if not out_of_bounds:
            self._enforce_boundaries()
    
    def _enforce_boundaries(self):
        """Keep ball and agents within field boundaries with collision response"""
        # Field boundaries: Center at (0,0), so field goes from -50 to +50 in X, -32.5 to +32.5 in Y
        min_x, max_x = -self.width / 2, self.width / 2
        min_y, max_y = -self.height / 2, self.height / 2
        
        # Ball boundaries - don't bounce, just keep in bounds (set pieces handle out of bounds)
        self.ball_pos[0] = np.clip(self.ball_pos[0], min_x + 0.1, max_x - 0.1)
        self.ball_pos[1] = np.clip(self.ball_pos[1], min_y + 0.1, max_y - 0.1)
        
        # Agent boundaries with collision response
        for agent in self.agents:
            # Handle boundary collisions for agents
            if agent.pos[0] <= min_x + self.agent_radius:
                agent.pos[0] = min_x + self.agent_radius
                if agent.vel[0] < 0:
                    agent.vel[0] *= -0.5  # Bounce back with damping
                    
            elif agent.pos[0] >= max_x - self.agent_radius:
                agent.pos[0] = max_x - self.agent_radius
                if agent.vel[0] > 0:
                    agent.vel[0] *= -0.5
            
            if agent.pos[1] <= min_y + self.agent_radius:
                agent.pos[1] = min_y + self.agent_radius
                if agent.vel[1] < 0:
                    agent.vel[1] *= -0.5
                    
            elif agent.pos[1] >= max_y - self.agent_radius:
                agent.pos[1] = max_y - self.agent_radius
                if agent.vel[1] > 0:
                    agent.vel[1] *= -0.5
    
    
    def _calculate_rewards(self, actions: List[Actions], goal_scored: bool) -> List[float]:
        """Calculate rewards for each agent including positional and tactical play"""
        rewards = []
        
        for agent, action in zip(self.agents, actions):
            reward = 0.0
            
            # Positional discipline rewards
            in_zone = False
            if hasattr(agent, 'is_in_zone') and hasattr(agent, 'distance_from_zone'):
                in_zone = agent.is_in_zone()
                distance_from_zone = agent.distance_from_zone()
                
                if in_zone:
                    # Reward for being in zone
                    reward += 1.0
                else:
                    # Penalty for being out of zone (stronger penalty for defenders/attackers)
                    if agent.role in ['defender', 'attacker']:
                        penalty = min(5.0, distance_from_zone * 0.3)
                        reward -= penalty
                    else:
                        reward -= distance_from_zone * 0.1
            
            # Role-specific positioning rewards
            if agent.role == 'defender':
                # Defenders rewarded for staying back
                if agent.team.name == 'BLUE':
                    if agent.pos[0] < -10.0:  # Staying in defensive half
                        reward += 0.5
                else:
                    if agent.pos[0] > 10.0:
                        reward += 0.5
                
                # Reward defender for passing when at zone boundary with ball
                if agent.has_ball and action == Actions.PASS:
                    if hasattr(agent, 'zone_center'):
                        if agent.team.name == 'BLUE':
                            at_boundary = agent.pos[0] > (agent.zone_center[0] + agent.zone_x_range * 0.7)
                        else:
                            at_boundary = agent.pos[0] < (agent.zone_center[0] - agent.zone_x_range * 0.7)
                        
                        if at_boundary:
                            reward += 5.0  # Big reward for passing forward from zone boundary
            
            elif agent.role == 'attacker':
                # Attackers rewarded for staying forward
                if agent.team.name == 'BLUE':
                    if agent.pos[0] > 10.0:  # Staying in attacking half
                        reward += 0.5
                else:
                    if agent.pos[0] < -10.0:
                        reward += 0.5
                
                # Reward attacker for being ready to receive (in zone, no ball)
                if not agent.has_ball and in_zone:
                    reward += 0.3
                    
                    # Extra reward if teammate has ball
                    team_has_ball = any(a.has_ball for a in self.agents if a.team == agent.team)
                    if team_has_ball:
                        reward += 0.5
            
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
            
            # Distance to ball - only reward if it's their job to get it
            distance_to_ball = np.linalg.norm(agent.pos - self.ball_pos)
            
            # Only midfielders and players without specific zones get ball proximity rewards
            # Defenders and attackers should focus on their zones
            if agent.role in ['midfielder', 'player']:
                reward += max(0, 3.0 - distance_to_ball * 0.1)
            elif agent.role in ['defender', 'attacker']:
                # Only reward if ball is in their zone
                if in_zone and distance_to_ball < 15.0:
                    reward += max(0, 2.0 - distance_to_ball * 0.1)
                # Small reward even if ball is nearby but they're in position
                elif in_zone:
                    reward += 0.2
            
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
            
            # STAY action penalties/rewards based on context
            if action == Actions.STAY:
                # Attackers staying in position when ball is with team = good
                if agent.role == 'attacker' and in_zone:
                    team_has_ball = any(a.has_ball and a != agent for a in self.agents if a.team == agent.team)
                    if team_has_ball:
                        reward += 0.3  # Reward for staying in position to receive
                    else:
                        reward -= 0.3
                # Defenders staying in position = good
                elif agent.role == 'defender' and in_zone:
                    reward += 0.2
                else:
                    reward -= 0.5  # Penalty for idle when should be active
            
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
    
    def _determine_last_touch(self):
        """Determine which team last touched the ball"""
        # Use tracked last touch if available
        if self.last_touch_team is not None:
            return self.last_touch_team

        if self.ball_owner:
            return self.ball_owner.team
        
        # Fallback: closest agent to ball as approximation
        if not self.agents:
            return None
        
        closest_agent = min(self.agents, key=lambda a: np.linalg.norm(a.pos - self.ball_pos))
        if np.linalg.norm(closest_agent.pos - self.ball_pos) < 5.0:
            return closest_agent.team
        
        return None
    
    def _trigger_corner_kick(self, team, is_left_side: bool, corner_top: bool):
        """Trigger a corner kick set piece"""
        from agents import Team
        
        self.set_piece_type = 'corner_kick'
        self.set_piece_team = team
        
        # Position ball at corner
        x = -self.width / 2 if is_left_side else self.width / 2
        y = (self.height / 2 - 1.0) if corner_top else -(self.height / 2 - 1.0)
        self.set_piece_position = np.array([x, y])
        self.ball_pos = self.set_piece_position.copy()
        self.ball_vel = np.zeros(2)
        self.ball_owner = None
        
        self.set_piece_timer = self.set_piece_duration
    
    def _trigger_goal_kick(self, team, is_left_side: bool):
        """Trigger a goal kick set piece"""
        from agents import Team
        
        self.set_piece_type = 'goal_kick'
        self.set_piece_team = team
        
        # Position ball in goal area
        x = -self.width / 2 + 6.0 if is_left_side else self.width / 2 - 6.0
        y = 0.0
        self.set_piece_position = np.array([x, y])
        self.ball_pos = self.set_piece_position.copy()
        self.ball_vel = np.zeros(2)
        self.ball_owner = None
        
        self.set_piece_timer = self.set_piece_duration
    
    def _trigger_throw_in(self, team, position: np.ndarray):
        """Trigger a throw-in set piece (Rule 15)"""
        from agents import Team
        
        self.set_piece_type = 'throw_in'
        self.set_piece_team = team
        self.set_piece_position = position.copy()
        self.ball_pos = self.set_piece_position.copy()
        self.ball_vel = np.zeros(2)
        self.ball_owner = None
        
        self.set_piece_timer = self.set_piece_duration
    
    def _handle_set_piece(self, actions: List[Actions]):
        """Handle set piece positioning and execution"""
        from agents import Team
        
        self.set_piece_timer -= 1
        
        if self.set_piece_timer > 0:
            # Positioning phase - move agents to appropriate positions
            self._position_agents_for_set_piece()
        else:
            # Execute set piece
            self._execute_set_piece(actions)
            
            # Resume normal play
            self.set_piece_type = None
            self.set_piece_team = None
            self.set_piece_position = None
    
    def _position_agents_for_set_piece(self):
        """Move agents to appropriate positions for set piece"""
        if self.set_piece_type == 'corner_kick' and self.set_piece_position is not None:
            # Position one attacking player near corner, others in penalty area
            attacking_team = [a for a in self.agents if a.team == self.set_piece_team]
            defending_team = [a for a in self.agents if a.team != self.set_piece_team]
            
            if attacking_team:
                # Closest attacker takes corner (slower movement: 0.15 from 0.3)
                corner_taker = min(attacking_team, key=lambda a: np.linalg.norm(a.pos - self.set_piece_position))
                target_pos = self.set_piece_position + np.array([2.0, 0.0] if self.set_piece_position[0] < 0 else [-2.0, 0.0])
                direction = target_pos - corner_taker.pos
                if np.linalg.norm(direction) > 0.5:
                    corner_taker.vel = direction * 0.15
                    corner_taker.pos += corner_taker.vel
                
                # Other attackers move to penalty area (slower: 0.1 from 0.2)
                if self.set_piece_team:
                    goal_x = self.width / 2 if self.set_piece_team.name == 'BLUE' else -self.width / 2
                    for i, attacker in enumerate(attacking_team[1:]):
                        target = np.array([goal_x - (5.0 if goal_x > 0 else -5.0), (i - len(attacking_team)//2) * 5.0])
                        direction = target - attacker.pos
                        if np.linalg.norm(direction) > 1.0:
                            attacker.vel = direction * 0.1
                            attacker.pos += attacker.vel
            
            # Defenders mark attackers (slower: 0.1 from 0.2)
            for defender in defending_team:
                if attacking_team:
                    nearest_attacker = min(attacking_team, key=lambda a: np.linalg.norm(a.pos - defender.pos))
                    direction = nearest_attacker.pos - defender.pos
                    if np.linalg.norm(direction) > 2.0:
                        defender.vel = direction * 0.1
                        defender.pos += defender.vel
        
        elif self.set_piece_type == 'throw_in' and self.set_piece_position is not None:
            # Position throwing player at sideline (slower: 0.15 from 0.3)
            throwing_team = [a for a in self.agents if a.team == self.set_piece_team]
            
            if throwing_team:
                thrower = min(throwing_team, key=lambda a: np.linalg.norm(a.pos - self.set_piece_position))
                direction = self.set_piece_position - thrower.pos
                if np.linalg.norm(direction) > 0.5:
                    thrower.vel = direction * 0.15
                    thrower.pos += thrower.vel
    
    def _execute_set_piece(self, actions: List[Actions]):
        """Execute the set piece"""
        if self.set_piece_type == 'corner_kick' and self.set_piece_position is not None:
            # Corner kick behaves similar to a throw-in: closest attacker takes it and passes to a teammate
            attacking_team = [a for a in self.agents if a.team == self.set_piece_team]
            if attacking_team and self.set_piece_team:
                # Corner taker: closest attacker to the corner spot
                corner_taker = min(attacking_team, key=lambda a: np.linalg.norm(a.pos - self.set_piece_position))

                # Select target teammate (exclude corner taker)
                teammates = [a for a in attacking_team if a is not corner_taker]
                if teammates:
                    # Prefer targets in/near the penalty area; otherwise nearest teammate
                    goal_x = self.width / 2 if self.set_piece_team.name == 'BLUE' else -self.width / 2
                    in_box = [a for a in teammates if (abs(goal_x - a.pos[0]) <= 20.0 and abs(a.pos[1]) <= (self.penalty_area_height / 2))]
                    if in_box:
                        target_teammate = min(in_box, key=lambda a: np.linalg.norm(a.pos - corner_taker.pos))
                    else:
                        target_teammate = min(teammates, key=lambda a: np.linalg.norm(a.pos - corner_taker.pos))

                    # Pass the ball from corner spot to the target teammate (throw-in like speed)
                    direction = target_teammate.pos - self.ball_pos
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        direction = direction / distance
                        pass_speed = min(7.0, 3.0 + 0.1 * distance)
                        self.ball_vel = direction * pass_speed
                        # Last touch is the corner taker's team
                        self.last_touch_team = corner_taker.team
                else:
                    # No available teammate: gently play the ball into the nearest penalty area
                    goal_x = self.width / 2 if self.set_piece_team.name == 'BLUE' else -self.width / 2
                    target = np.array([goal_x - (10.0 if goal_x > 0 else -10.0), random.uniform(-5.0, 5.0)])
                    direction = target - self.ball_pos
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        direction = direction / distance
                        self.ball_vel = direction * 6.0
                        self.last_touch_team = corner_taker.team
        
        elif self.set_piece_type == 'throw_in' and self.set_piece_position is not None:
            # Find thrower
            throwing_team = [a for a in self.agents if a.team == self.set_piece_team]
            if throwing_team:
                thrower = min(throwing_team, key=lambda a: np.linalg.norm(a.pos - self.set_piece_position))
                
                # Throw ball to nearest teammate
                teammates = [a for a in throwing_team if a != thrower]
                if teammates:
                    target_teammate = min(teammates, key=lambda a: np.linalg.norm(a.pos - thrower.pos))
                    direction = target_teammate.pos - self.ball_pos
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        direction = direction / distance
                        # Throw speed scales with distance (closer = softer)
                        throw_speed = min(6.0, 3.0 + distance * 0.1)
                        self.ball_vel = direction * throw_speed
        
        elif self.set_piece_type == 'goal_kick' and self.set_piece_position is not None:
            # Goalkeeper or defender kicks ball upfield
            kicking_team = [a for a in self.agents if a.team == self.set_piece_team]
            if kicking_team:
                kicker = min(kicking_team, key=lambda a: np.linalg.norm(a.pos - self.set_piece_position))
                
                # Kick ball towards midfield with variation
                target_x = random.uniform(-10.0, 10.0)
                target_y = random.uniform(-self.height / 4, self.height / 4)
                direction = np.array([target_x, target_y]) - self.ball_pos
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    self.ball_vel = direction * 10.0

