#!/usr/bin/env python3
"""
Pygame-based visualization system for soccer simulation.
Supports reading checkpoint data and visualizing training progress and gameplay.
"""

import pygame
import numpy as np
import pickle
import json
import os
import time
from typing import Optional, Dict, List, Tuple
import argparse

# Import our modules
from agents import Defender, Attacker, Team, FieldDistribution
from env import Environment
from bdi import Actions
from main import TrainingConfig, TrainingStats, load_checkpoint

# Pygame constants
WIDTH = 1200
HEIGHT = 800
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
BLUE = (0, 100, 255)
LIGHT_BLUE = (173, 216, 230)
RED = (255, 0, 0)
LIGHT_RED = (255, 182, 193)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (211, 211, 211)

class SoccerVisualizer:
    """
    Pygame-based visualizer for soccer simulation with checkpoint support.
    """
    
    def __init__(self, width=WIDTH, height=HEIGHT):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Soccer Simulation Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 36)
        
        # Visualization settings
        self.simulation_speed = 1.0  # Speed multiplier (1.0 = normal, 0.5 = half speed)
        self.show_trails = True
        self.show_beliefs = True
        self.show_stats = True
        self.paused = False
        
        # Field scaling
        self.field_margin = 50
        self.field_width = width - 2 * self.field_margin
        self.field_height = height - 2 * self.field_margin - 100  # Space for UI
        
        # Agent trails for visualization
        self.agent_trails = {}
        self.max_trail_length = 30
        
        # Current simulation state
        self.env = None
        self.agents = []
        self.current_episode = 0
        self.total_episodes = 0
        self.stats = None
        
    def world_to_screen(self, pos: np.ndarray) -> Tuple[int, int]:
        """
        Convert world coordinates to screen coordinates.
        World: field center at (0,0), X=[-50,50], Y=[-32.5,32.5]
        Screen: (0,0) at top-left
        """
        # World coordinate system: center at (0,0)
        world_x, world_y = pos[0], pos[1]
        
        # Scale to screen
        screen_x = self.field_margin + (world_x + 50) * (self.field_width / 100)
        screen_y = self.field_margin + (-world_y + 32.5) * (self.field_height / 65)
        
        return int(screen_x), int(screen_y)
    
    def draw_field(self):
        """Draw the soccer field with proper markings."""
        # Fill background
        self.screen.fill(DARK_GREEN)
        
        # Draw field background
        field_rect = pygame.Rect(self.field_margin, self.field_margin, 
                                self.field_width, self.field_height)
        pygame.draw.rect(self.screen, GREEN, field_rect)
        
        # Field boundaries
        pygame.draw.rect(self.screen, WHITE, field_rect, 3)
        
        # Center line
        center_x = self.field_margin + self.field_width // 2
        pygame.draw.line(self.screen, WHITE, 
                        (center_x, self.field_margin), 
                        (center_x, self.field_margin + self.field_height), 2)
        
        # Center circle
        center_y = self.field_margin + self.field_height // 2
        pygame.draw.circle(self.screen, WHITE, (center_x, center_y), 60, 2)
        
        # Goals
        goal_width = int(7.32 * (self.field_height / 65))  # Scale goal width
        goal_height = 20
        
        # Left goal (Blue defends)
        left_goal_y = center_y - goal_width // 2
        left_goal_rect = pygame.Rect(self.field_margin - goal_height, left_goal_y, 
                                    goal_height, goal_width)
        pygame.draw.rect(self.screen, WHITE, left_goal_rect, 2)
        
        # Right goal (White defends)
        right_goal_rect = pygame.Rect(self.field_margin + self.field_width, left_goal_y, 
                                     goal_height, goal_width)
        pygame.draw.rect(self.screen, WHITE, right_goal_rect, 2)
        
        # Penalty areas
        penalty_width = int(16.5 * (self.field_width / 100))
        penalty_height = int(40.3 * (self.field_height / 65))
        penalty_y = center_y - penalty_height // 2
        
        # Left penalty area
        left_penalty = pygame.Rect(self.field_margin, penalty_y, penalty_width, penalty_height)
        pygame.draw.rect(self.screen, WHITE, left_penalty, 2)
        
        # Right penalty area
        right_penalty = pygame.Rect(self.field_margin + self.field_width - penalty_width, 
                                   penalty_y, penalty_width, penalty_height)
        pygame.draw.rect(self.screen, WHITE, right_penalty, 2)
    
    def draw_ball(self, pos: np.ndarray, vel: np.ndarray):
        """Draw the ball with motion indicators."""
        screen_pos = self.world_to_screen(pos)
        
        # Ball shadow
        shadow_pos = (screen_pos[0] + 2, screen_pos[1] + 2)
        pygame.draw.circle(self.screen, GRAY, shadow_pos, 8)
        
        # Ball
        pygame.draw.circle(self.screen, WHITE, screen_pos, 8)
        pygame.draw.circle(self.screen, BLACK, screen_pos, 8, 2)
        
        # Velocity indicator
        if np.linalg.norm(vel) > 0.1:
            vel_end = pos + vel * 2  # Scale velocity for visualization
            vel_screen_end = self.world_to_screen(vel_end)
            pygame.draw.line(self.screen, YELLOW, screen_pos, vel_screen_end, 2)
    
    def draw_agent(self, agent, index: int):
        """Draw an agent with team colors and role indicators."""
        screen_pos = self.world_to_screen(agent.pos)
        
        # Update trail
        if agent not in self.agent_trails:
            self.agent_trails[agent] = []
        
        self.agent_trails[agent].append(screen_pos)
        if len(self.agent_trails[agent]) > self.max_trail_length:
            self.agent_trails[agent].pop(0)
        
        # Draw trail if enabled
        if self.show_trails and len(self.agent_trails[agent]) > 1:
            trail_color = LIGHT_BLUE if agent.team == Team.BLUE else LIGHT_RED
            for i in range(1, len(self.agent_trails[agent])):
                alpha = i / len(self.agent_trails[agent])
                start = self.agent_trails[agent][i-1]
                end = self.agent_trails[agent][i]
                pygame.draw.line(self.screen, trail_color, start, end, max(1, int(alpha * 3)))
        
        # Agent colors based on team
        if agent.team == Team.BLUE:
            color = BLUE
            text_color = WHITE
        else:
            color = WHITE
            text_color = BLACK
        
        # Agent body
        radius = 15
        pygame.draw.circle(self.screen, color, screen_pos, radius)
        pygame.draw.circle(self.screen, BLACK, screen_pos, radius, 2)
        
        # Ball possession indicator
        if hasattr(agent, 'has_ball') and agent.has_ball:
            pygame.draw.circle(self.screen, YELLOW, screen_pos, radius + 5, 3)
        
        # Role indicator
        role_text = agent.role[0].upper()  # First letter of role
        text_surface = self.small_font.render(role_text, True, text_color)
        text_rect = text_surface.get_rect(center=screen_pos)
        self.screen.blit(text_surface, text_rect)
        
        # Velocity vector
        if hasattr(agent, 'vel') and np.linalg.norm(agent.vel) > 0.1:
            vel_end = agent.pos + agent.vel * 3
            vel_screen_end = self.world_to_screen(vel_end)
            pygame.draw.line(self.screen, color, screen_pos, vel_screen_end, 2)
        
        # Agent beliefs visualization (if enabled)
        if self.show_beliefs and hasattr(agent, 'beliefs'):
            self.draw_agent_beliefs(agent, screen_pos)
    
    def draw_agent_beliefs(self, agent, screen_pos: Tuple[int, int]):
        """Draw agent beliefs as text near the agent."""
        if not hasattr(agent, 'beliefs') or not self.env:
            return
            
        beliefs = self.env.get_beliefs(agent)
        
        # Display key beliefs
        belief_lines = [
            f"Ball: {beliefs['distance_to_ball']:.1f}m",
            f"Goal: {beliefs['distance_to_goal']:.1f}m",
            f"Opp: {beliefs['distance_to_opponent']:.1f}m"
        ]
        
        y_offset = -40
        for line in belief_lines:
            text_surface = self.small_font.render(line, True, BLACK)
            text_pos = (screen_pos[0] + 20, screen_pos[1] + y_offset)
            
            # Background for readability
            text_rect = text_surface.get_rect(topleft=text_pos)
            text_rect.inflate_ip(4, 2)
            pygame.draw.rect(self.screen, WHITE, text_rect)
            pygame.draw.rect(self.screen, BLACK, text_rect, 1)
            
            self.screen.blit(text_surface, text_pos)
            y_offset += 15
    
    def draw_ui(self):
        """Draw user interface elements."""
        ui_y = self.field_margin + self.field_height + 10
        
        # Score
        if self.env:
            score_text = f"Score - Blue: {self.env.score['blue']}  White: {self.env.score['white']}"
            score_surface = self.large_font.render(score_text, True, BLACK)
            self.screen.blit(score_surface, (self.field_margin, ui_y))
        
        # Episode info
        if self.total_episodes > 0:
            episode_text = f"Episode: {self.current_episode}/{self.total_episodes}"
            episode_surface = self.font.render(episode_text, True, BLACK)
            self.screen.blit(episode_surface, (self.field_margin, ui_y + 40))
        
        # Speed control
        speed_text = f"Speed: {self.simulation_speed:.1f}x {'(PAUSED)' if self.paused else ''}"
        speed_surface = self.font.render(speed_text, True, BLACK)
        self.screen.blit(speed_surface, (self.field_margin + 300, ui_y + 40))
        
        # Controls
        controls = [
            "Controls: SPACE=Pause, +/- Speed, T=Trails, B=Beliefs, S=Stats, Q=Quit"
        ]
        
        control_y = ui_y + 65
        for control in controls:
            control_surface = self.small_font.render(control, True, BLACK)
            self.screen.blit(control_surface, (self.field_margin, control_y))
            control_y += 20
        
        # Statistics (if enabled and available)
        if self.show_stats and self.stats:
            self.draw_statistics()
    
    def draw_statistics(self):
        """Draw training statistics."""
        if not self.stats or not self.stats.episode_rewards:
            return
            
        stats_x = self.width - 250
        stats_y = self.field_margin
        
        # Background
        stats_rect = pygame.Rect(stats_x - 10, stats_y - 10, 240, 200)
        pygame.draw.rect(self.screen, WHITE, stats_rect)
        pygame.draw.rect(self.screen, BLACK, stats_rect, 2)
        
        # Title
        title_surface = self.font.render("Training Stats", True, BLACK)
        self.screen.blit(title_surface, (stats_x, stats_y))
        
        # Stats
        summary = self.stats.get_summary_stats()
        stat_lines = [
            f"Episodes: {summary.get('episodes_completed', 0)}",
            f"Avg Reward: {summary.get('avg_reward', 0):.1f}",
            f"Win Rate: {summary.get('win_rate', 0):.1%}",
            f"Best Win Rate: {summary.get('best_win_rate', 0):.1%}",
            f"Exploration: {summary.get('current_exploration_rate', 0):.3f}",
            f"Q-States: {summary.get('avg_q_table_size', 0):.0f}",
            f"Time: {summary.get('total_training_time', 0)/60:.1f}min"
        ]
        
        y_offset = 25
        for line in stat_lines:
            stat_surface = self.small_font.render(line, True, BLACK)
            self.screen.blit(stat_surface, (stats_x, stats_y + y_offset))
            y_offset += 18
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.simulation_speed = min(5.0, self.simulation_speed + 0.1)
                elif event.key == pygame.K_MINUS:
                    self.simulation_speed = max(0.1, self.simulation_speed - 0.1)
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                elif event.key == pygame.K_b:
                    self.show_beliefs = not self.show_beliefs
                elif event.key == pygame.K_s:
                    self.show_stats = not self.show_stats
        return True
    
    def load_checkpoint_and_setup(self, checkpoint_path: str) -> bool:
        """Load checkpoint data and setup visualization."""
        try:
            # Create dummy config to load checkpoint
            dummy_config = FieldDistribution()
            
            # Load checkpoint
            episode, stats, training_config = load_checkpoint(checkpoint_path, dummy_config)
            
            print(f"Loaded checkpoint from episode {episode}")
            print(f"Training config: {training_config.__dict__}")
            
            # Setup environment and agents from checkpoint
            self.env = Environment()
            self.stats = stats
            self.current_episode = episode
            self.total_episodes = training_config.num_episodes
            
            # Load checkpoint data to get agent information
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Create agents based on checkpoint data
            self.agents = []
            for i, agent_data in enumerate(checkpoint_data['agents']):
                team = Team.BLUE if agent_data['team'] == 'BLUE' else Team.WHITE
                pos = np.array(agent_data['position'])
                
                if agent_data['role'] == 'attacker':
                    agent = Attacker(self.env, team, pos=pos)
                elif agent_data['role'] == 'defender':
                    agent = Defender(self.env, team, pos=pos)
                else:
                    agent = Attacker(self.env, team, pos=pos)  # Default
                
                # Restore Q-learning state
                for key, value in agent_data['q_table'].items():
                    agent.q_policy.Q[key] = value
                
                params = agent_data['q_policy_params']
                agent.q_policy.alpha = params['alpha']
                agent.q_policy.gamma = params['gamma']
                agent.q_policy.eps = params['eps']
                
                self.agents.append(agent)
            
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_simulation_episode(self, max_steps: int = 600):
        """Run a single episode of simulation for visualization."""
        if not self.env or not self.agents:
            return
        
        # Reset environment
        self.env.reset()
        for agent in self.agents:
            agent.reset_episode()
        
        step_count = 0
        last_time = time.time()
        
        running = True
        while running and step_count < max_steps:
            # Handle events
            if not self.handle_events():
                return False
            
            # Control simulation speed
            current_time = time.time()
            if not self.paused and (current_time - last_time) >= (0.05 / self.simulation_speed):  # 50ms timestep
                # Get actions from agents
                actions = []
                for agent in self.agents:
                    action = agent.act()
                    actions.append(action)
                
                # Step environment
                observations, rewards, done, info = self.env.step(actions)
                
                # Update agents
                for agent, reward in zip(self.agents, rewards):
                    agent.learn(reward, done)
                
                step_count += 1
                last_time = current_time
                
                if done:
                    break
            
            # Draw everything
            self.draw_field()
            
            # Draw ball
            self.draw_ball(self.env.ball_pos, self.env.ball_vel)
            
            # Draw agents
            for i, agent in enumerate(self.agents):
                self.draw_agent(agent, i)
            
            # Draw UI
            self.draw_ui()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        
        return True
    
    def run_continuous_visualization(self):
        """Run continuous visualization with multiple episodes."""
        episode_count = 0
        
        while True:
            print(f"Starting episode {episode_count + 1}")
            
            if not self.run_simulation_episode():
                break  # User quit
            
            episode_count += 1
            
            # Small delay between episodes
            time.sleep(1.0)
    
    def close(self):
        """Clean up pygame."""
        pygame.quit()


def list_checkpoints(checkpoint_dir: str) -> List[str]:
    """List available checkpoint files."""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pkl') and 'checkpoint' in filename:
            checkpoints.append(os.path.join(checkpoint_dir, filename))
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoints


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Soccer Simulation Visualizer')
    parser.add_argument('--checkpoint', '-c', type=str, help='Path to checkpoint file')
    parser.add_argument('--checkpoint-dir', '-d', type=str, default='checkpoints', 
                       help='Directory containing checkpoints')
    parser.add_argument('--list', '-l', action='store_true', 
                       help='List available checkpoints')
    
    args = parser.parse_args()
    
    # List checkpoints if requested
    if args.list:
        checkpoints = list_checkpoints(args.checkpoint_dir)
        if checkpoints:
            print("Available checkpoints:")
            for i, checkpoint in enumerate(checkpoints):
                print(f"  {i+1}: {os.path.basename(checkpoint)}")
        else:
            print("No checkpoints found.")
        return
    
    # Get checkpoint to use
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoints = list_checkpoints(args.checkpoint_dir)
        if checkpoints:
            checkpoint_path = checkpoints[0]  # Use most recent
            print(f"Using most recent checkpoint: {os.path.basename(checkpoint_path)}")
        else:
            print("No checkpoints found. Run training first.")
            return
    
    # Create and run visualizer
    visualizer = SoccerVisualizer()
    
    try:
        if visualizer.load_checkpoint_and_setup(checkpoint_path):
            print("Checkpoint loaded successfully. Starting visualization...")
            print("\nControls:")
            print("  SPACE - Pause/Resume")
            print("  +/-   - Adjust speed")
            print("  T     - Toggle trails")
            print("  B     - Toggle beliefs")
            print("  S     - Toggle statistics")
            print("  Q     - Quit")
            
            visualizer.run_continuous_visualization()
        else:
            print("Failed to load checkpoint.")
    
    except KeyboardInterrupt:
        print("\nVisualization interrupted.")
    
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
