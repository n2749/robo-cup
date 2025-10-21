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
from agents import Defender, Attacker, Goalkeeper, Team, FieldDistribution
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
        self.show_beliefs = False
        self.show_stats = True
        self.paused = False
        
        # Field scaling
        self.field_margin = 50
        self.field_width = width - 2 * self.field_margin
        self.field_height = height - 2 * self.field_margin - 100  # Space for UI
        
        # Agent trails for visualization
        self.agent_trails = {}
        self.max_trail_length = 30
        
        # Agent selection
        self.selected_agent = None
        self.show_agent_details = False
        
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
        
        # Corner arcs (visual markers for corner kicks)
        corner_radius = 15
        # Top-left corner
        pygame.draw.circle(self.screen, WHITE, (self.field_margin, self.field_margin), corner_radius, 2)
        # Top-right corner
        pygame.draw.circle(self.screen, WHITE, (self.field_margin + self.field_width, self.field_margin), corner_radius, 2)
        # Bottom-left corner
        pygame.draw.circle(self.screen, WHITE, (self.field_margin, self.field_margin + self.field_height), corner_radius, 2)
        # Bottom-right corner
        pygame.draw.circle(self.screen, WHITE, (self.field_margin + self.field_width, self.field_margin + self.field_height), corner_radius, 2)
    
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
        
        # Highlight selected agent
        is_selected = (self.selected_agent == agent)
        if is_selected:
            pygame.draw.circle(self.screen, YELLOW, screen_pos, 25, 4)
            pygame.draw.circle(self.screen, ORANGE, screen_pos, 22, 2)
        
        # Agent body
        radius = 15
        pygame.draw.circle(self.screen, color, screen_pos, radius)
        pygame.draw.circle(self.screen, BLACK, screen_pos, radius, 2 if not is_selected else 3)
        
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
            
            # Set piece indicator
            if hasattr(self.env, 'set_piece_type') and self.env.set_piece_type:
                set_piece_text = f"SET PIECE: {self.env.set_piece_type.upper().replace('_', ' ')}"
                if hasattr(self.env, 'set_piece_team') and self.env.set_piece_team:
                    set_piece_text += f" - {self.env.set_piece_team.name}"
                set_piece_surface = self.font.render(set_piece_text, True, ORANGE)
                self.screen.blit(set_piece_surface, (self.field_margin + 500, ui_y))
        
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
            "Controls: SPACE=Pause, +/- Speed, T=Trails, B=Beliefs, S=Stats, D=Details, Q=Quit",
            "Click on agent to select and view details (works best when paused)"
        ]
        
        control_y = ui_y + 65
        for control in controls:
            control_surface = self.small_font.render(control, True, BLACK)
            self.screen.blit(control_surface, (self.field_margin, control_y))
            control_y += 18
        
        # Selected agent indicator
        if self.selected_agent:
            agent_info = f"Selected: {self.selected_agent.role} ({self.selected_agent.team.name})"
            info_surface = self.small_font.render(agent_info, True, ORANGE)
            self.screen.blit(info_surface, (self.field_margin + 600, ui_y + 40))
        
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
    
    def draw_agent_details_panel(self):
        """Draw detailed information panel for selected agent"""
        if not self.selected_agent or not self.show_agent_details:
            return
        
        agent = self.selected_agent
        
        # Panel dimensions and position
        panel_width = 400
        panel_height = self.height - 100
        panel_x = 10
        panel_y = 10
        
        # Background with border
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, WHITE, panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 3)
        
        # Current Y position for text
        y = panel_y + 10
        x_margin = panel_x + 10
        
        # Title
        title_color = BLUE if agent.team == Team.BLUE else RED
        title_text = f"{agent.role.upper()} - {agent.team.name}"
        title_surface = self.font.render(title_text, True, title_color)
        self.screen.blit(title_surface, (x_margin, y))
        y += 30
        
        # Position info
        pos_text = f"Position: ({agent.pos[0]:.1f}, {agent.pos[1]:.1f})"
        vel_magnitude = np.linalg.norm(agent.vel) if hasattr(agent, 'vel') else 0
        vel_text = f"Velocity: {vel_magnitude:.2f} m/s"
        
        self.screen.blit(self.small_font.render(pos_text, True, BLACK), (x_margin, y))
        y += 18
        self.screen.blit(self.small_font.render(vel_text, True, BLACK), (x_margin, y))
        y += 18
        self.screen.blit(self.small_font.render(f"Has Ball: {agent.has_ball}", True, BLACK), (x_margin, y))
        y += 25
        
        # Beliefs section
        if hasattr(agent, 'beliefs') and self.env:
            beliefs = self.env.get_beliefs(agent)
            
            section_surface = self.font.render("BELIEFS", True, DARK_GREEN)
            self.screen.blit(section_surface, (x_margin, y))
            y += 25
            
            belief_lines = [
                f"Distance to Ball: {beliefs.get('distance_to_ball', 0):.1f}m",
                f"Distance to Goal: {beliefs.get('distance_to_goal', 0):.1f}m",
                f"Distance to Home: {beliefs.get('distance_to_home_goal', 0):.1f}m",
                f"Distance to Opponent: {beliefs.get('distance_to_opponent', 0):.1f}m",
                f"Teammate Open: {beliefs.get('teammate_open', False)}",
                f"Goal Open: {beliefs.get('goal_open', False)}",
            ]
            
            if hasattr(agent, 'beliefs'):
                belief_lines.extend([
                    f"Has Possession: {agent.beliefs.has_ball_possession}",
                    f"Team Has Ball: {agent.beliefs.team_has_possession}",
                    f"In Attack Third: {agent.beliefs.in_attacking_third}",
                    f"In Defense Third: {agent.beliefs.in_defensive_third}",
                    f"Opponent Threat: {agent.beliefs.opponent_threatening}",
                ])
            
            for line in belief_lines:
                self.screen.blit(self.small_font.render(line, True, BLACK), (x_margin, y))
                y += 16
            
            y += 10
        
        # Desires section
        if hasattr(agent, 'desires'):
            section_surface = self.font.render("DESIRES", True, DARK_GREEN)
            self.screen.blit(section_surface, (x_margin, y))
            y += 25
            
            desires = agent.desires.get_current_desires_summary()
            desire_lines = [
                f"{name.capitalize()}: {value:.2f}"
                for name, value in desires.items()
            ]
            
            for line in desire_lines:
                self.screen.blit(self.small_font.render(line, True, BLACK), (x_margin, y))
                y += 16
            
            y += 10
        
        # Pass prediction section
        if hasattr(agent, 'last_pass_prediction'):
            prediction = agent.last_pass_prediction
            if prediction.get('probability') is not None:
                section_surface = self.font.render("PASS PREDICT", True, DARK_GREEN)
                self.screen.blit(section_surface, (x_margin, y))
                y += 25
                
                prob = prediction['probability'] * 100.0
                confidence_label = prediction.get('confidence_label', 'Unknown')
                confidence_score = prediction.get('confidence_score', 0.0)
                target_role = prediction.get('features', {}).get('target_role', 'n/a')
                pass_type = prediction.get('features', {}).get('pass_type', 'n/a')
                receiver_skill = prediction.get('features', {}).get('receiver_skill')
                calibration_error = getattr(agent.env, 'pass_calibration_error', 0.5) if hasattr(agent, 'env') else 0.5
                pass_success_rate = (agent.pass_successes / agent.pass_attempts) * 100.0 if agent.pass_attempts else 0.0
                receive_success_rate = (agent.receive_successes / agent.receive_attempts) * 100.0 if agent.receive_attempts else 0.0
                lines = [
                    f"Prob: {prob:.1f}%",
                    f"Confidence: {confidence_label} ({confidence_score:.2f})",
                    f"Target Role: {target_role}",
                    f"Pass Type: {pass_type}",
                    f"Skill: {getattr(agent, 'pass_skill', 1.0):.2f}",
                    f"Recv Skill: {getattr(agent, 'receive_skill', 1.0):.2f}",
                    f"Pass SR: {pass_success_rate:.0f}% ({agent.pass_successes}/{agent.pass_attempts})",
                    f"Recv SR: {receive_success_rate:.0f}% ({agent.receive_successes}/{agent.receive_attempts})",
                    f"Calib Err: {calibration_error:.2f}",
                ]
                
                for line in lines:
                    self.screen.blit(self.small_font.render(line, True, BLACK), (x_margin, y))
                    y += 16
                
                y += 10
        
        # Intentions section
        if hasattr(agent, 'intentions'):
            section_surface = self.font.render("INTENTIONS", True, DARK_GREEN)
            self.screen.blit(section_surface, (x_margin, y))
            y += 25
            
            current_action = agent.intentions.current_action
            commitment = agent.intentions.commitment_strength
            
            action_text = f"Current Action: {current_action.name if current_action else 'None'}"
            commit_text = f"Commitment: {commitment:.2f}"
            
            self.screen.blit(self.small_font.render(action_text, True, BLACK), (x_margin, y))
            y += 16
            self.screen.blit(self.small_font.render(commit_text, True, BLACK), (x_margin, y))
            y += 25
        
        # Q-Learning stats
        if hasattr(agent, 'q_policy'):
            section_surface = self.font.render("Q-LEARNING", True, DARK_GREEN)
            self.screen.blit(section_surface, (x_margin, y))
            y += 25
            
            stats = agent.q_policy.get_stats()
            q_lines = [
                f"Epsilon: {stats.get('epsilon', 0):.3f}",
                f"Q-Table Size: {stats.get('q_table_size', 0)}",
                f"Episodes: {stats.get('episode_count', 0)}",
            ]
            
            if hasattr(agent, 'episode_rewards') and agent.episode_rewards:
                recent = agent.episode_rewards[-10:]
                q_lines.append(f"Avg Recent Reward: {np.mean(recent):.2f}")
            
            for line in q_lines:
                self.screen.blit(self.small_font.render(line, True, BLACK), (x_margin, y))
                y += 16
            
            y += 10
        
        # Recent actions
        if hasattr(agent, 'actions_taken') and agent.actions_taken:
            section_surface = self.font.render("RECENT ACTIONS", True, DARK_GREEN)
            self.screen.blit(section_surface, (x_margin, y))
            y += 25
            
            recent_actions = agent.actions_taken[-5:]
            for action in recent_actions:
                action_name = action.name if hasattr(action, 'name') else str(action)
                self.screen.blit(self.small_font.render(f"- {action_name}", True, BLACK), (x_margin, y))
                y += 16
        
        # Close button hint
        hint_text = "Press ESC to close | Click agent to select"
        hint_surface = self.small_font.render(hint_text, True, GRAY)
        self.screen.blit(hint_surface, (x_margin, panel_y + panel_height - 20))
    
    def draw_set_piece_marker(self, position: np.ndarray, set_piece_type: str):
        """Draw a visual marker for set pieces on the field."""
        screen_pos = self.world_to_screen(position)
        
        if set_piece_type == 'corner_kick':
            # Draw pulsing circle for corner kick
            radius = 25 + int(5 * np.sin(pygame.time.get_ticks() / 200))
            pygame.draw.circle(self.screen, ORANGE, screen_pos, radius, 4)
            pygame.draw.circle(self.screen, YELLOW, screen_pos, radius - 5, 2)
            
            # Draw "CK" text
            text = self.font.render("CK", True, ORANGE)
            text_rect = text.get_rect(center=screen_pos)
            self.screen.blit(text, text_rect)
            
        elif set_piece_type == 'throw_in':
            # Draw arrow pointing inward from sideline
            pygame.draw.circle(self.screen, BLUE if position[1] > 0 else RED, screen_pos, 20, 4)
            
            # Draw arrow
            arrow_start = screen_pos
            arrow_end = (screen_pos[0], screen_pos[1] + (30 if position[1] > 0 else -30))
            pygame.draw.line(self.screen, ORANGE, arrow_start, arrow_end, 3)
            
            # Arrowhead
            arrow_tip = arrow_end
            left_point = (arrow_tip[0] - 8, arrow_tip[1] + (-8 if position[1] > 0 else 8))
            right_point = (arrow_tip[0] + 8, arrow_tip[1] + (-8 if position[1] > 0 else 8))
            pygame.draw.polygon(self.screen, ORANGE, [arrow_tip, left_point, right_point])
            
            # Draw "TI" text
            text = self.small_font.render("THROW-IN", True, ORANGE)
            text_rect = text.get_rect(center=(screen_pos[0], screen_pos[1] - 30))
            self.screen.blit(text, text_rect)
            
        elif set_piece_type == 'goal_kick':
            # Draw rectangle in goal area
            pygame.draw.rect(self.screen, LIGHT_GRAY, 
                           (screen_pos[0] - 25, screen_pos[1] - 15, 50, 30), 3)
            
            # Draw "GK" text
            text = self.font.render("GK", True, WHITE)
            text_rect = text.get_rect(center=screen_pos)
            # Background for text
            bg_rect = text_rect.inflate(10, 5)
            pygame.draw.rect(self.screen, DARK_GREEN, bg_rect)
            self.screen.blit(text, text_rect)
    
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
                elif event.key == pygame.K_d:
                    self.show_agent_details = not self.show_agent_details
                elif event.key == pygame.K_ESCAPE:
                    self.selected_agent = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_agent_click(event.pos)
        return True
    
    def _handle_agent_click(self, mouse_pos: Tuple[int, int]):
        """Handle mouse click to select agent"""
        if not self.agents:
            return
        
        # Find agent closest to click position
        min_distance = 20  # Maximum click distance in pixels
        clicked_agent = None
        
        for agent in self.agents:
            screen_pos = self.world_to_screen(agent.pos)
            distance = np.sqrt((mouse_pos[0] - screen_pos[0])**2 + (mouse_pos[1] - screen_pos[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                clicked_agent = agent
        
        if clicked_agent:
            self.selected_agent = clicked_agent
            self.show_agent_details = True
            print(f"Selected agent: {clicked_agent.role} ({clicked_agent.team.name})")
    
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
                pos = np.array(agent_data['pos'])
                base_pos = np.array(agent_data['base_pos'])
                
                if agent_data['role'] == 'attacker':
                    agent = Attacker(self.env, team, pos=pos, base_pos=base_pos)
                elif agent_data['role'] == 'defender':
                    agent = Defender(self.env, team, pos=pos, base_pos=base_pos)
                elif agent_data['role'] == 'goalkeeper':
                    agent = Goalkeeper(self.env, team, pos=pos, base_pos=base_pos)
                else:
                    agent = Attacker(self.env, team, pos=pos, base_pos=base_pos)  # Default
                
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
            
            # Draw set piece indicator on field
            if hasattr(self.env, 'set_piece_type') and self.env.set_piece_type and hasattr(self.env, 'set_piece_position'):
                if self.env.set_piece_position is not None:
                    self.draw_set_piece_marker(self.env.set_piece_position, self.env.set_piece_type)
            
            # Draw ball
            self.draw_ball(self.env.ball_pos, self.env.ball_vel)
            
            # Draw agents
            for i, agent in enumerate(self.agents):
                self.draw_agent(agent, i)
            
            # Draw UI
            self.draw_ui()
            
            # Draw agent details panel (on top of everything)
            if self.selected_agent and self.show_agent_details:
                self.draw_agent_details_panel()
            
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
