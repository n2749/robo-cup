#!/usr/bin/env python3
"""
Torch-based batched soccer environment for GA evaluation (experimental).

This is a simplified, vectorized environment implemented with PyTorch tensors to
run B batched matches in parallel on CPU or GPU. It focuses on enough physics and
rules to provide meaningful GA evaluation:
- 2D positions/velocities for agents and ball
- Possession via proximity
- MOVE/PASS/SHOOT/TACKLE/BLOCK/STAY actions (simplified semantics)
- Goal detection and scoring

It does NOT replicate all details of src/env.py (set pieces, collisions, spacing
rewards). The goal is to enable fast population evaluation.
"""
from __future__ import annotations

from typing import Dict, Tuple
import torch

from bdi import Actions
from team_config import get_available_formations, Formation442, Formation433, Formation352


class TorchSoccerEnv:
    def __init__(
        self,
        batch_size: int = 32,
        device: str = "cpu",
        field_width: float = 100.0,
        field_height: float = 65.0,
        max_steps: int = 600,
        possession_distance: float = 2.0,
        goal_width: float = 7.32,
        K1: float = 0.05,
        K2: float = 0.12,
        max_force: float = 6.0,
        ball_friction: float = 0.06,
    ):
        self.B = batch_size
        self.device = torch.device(device)

        # Field
        self.width = field_width
        self.height = field_height
        self.max_steps = max_steps
        self.goal_width = goal_width
        self.possession_distance = possession_distance
        self.K1 = K1
        self.K2 = K2
        self.max_force = max_force
        self.ball_friction = ball_friction

        # Team constants
        self.team_blue = 0
        self.team_white = 1

        # These will be set in build_from_formations()
        self.N = 0
        self.blue_mask = None  # [N]
        self.white_mask = None  # [N]
        self.initial_pos = None  # [N,2]

        # Runtime tensors
        self.pos = None         # [B,N,2]
        self.vel = None         # [B,N,2]
        self.has_ball = None    # [B,N] bool
        self.ball_pos = None    # [B,2]
        self.ball_vel = None    # [B,2]
        self.score_blue = None  # [B]
        self.score_white = None # [B]
        self.step_count = None  # [B]
        self.done = None        # [B] bool

    def build_from_formations(self, blue_form: str = "4-4-2", white_form: str = "4-4-2"):
        formations = get_available_formations()
        b_form = formations.get(blue_form, Formation442())
        w_form = formations.get(white_form, Formation442())

        # Get positions (role, pos) lists for each team
        blue_positions = b_form.get_positions(
            team=type("T", (), {"name": "BLUE"}), field_width=self.width, field_height=self.height
        )
        white_positions = w_form.get_positions(
            team=type("T", (), {"name": "WHITE"}), field_width=self.width, field_height=self.height
        )

        # Assemble initial positions [N,2] in team order (all blue then all white)
        all_positions = []
        all_teams = []
        for role, p in blue_positions:
            all_positions.append(torch.tensor(p, dtype=torch.float32))
            all_teams.append(self.team_blue)
        for role, p in white_positions:
            all_positions.append(torch.tensor(p, dtype=torch.float32))
            all_teams.append(self.team_white)

        self.N = len(all_positions)
        self.initial_pos = torch.stack(all_positions, dim=0).to(self.device)  # [N,2]

        teams = torch.tensor(all_teams, dtype=torch.long, device=self.device)  # [N]
        self.blue_mask = teams == self.team_blue
        self.white_mask = teams == self.team_white

        # Prepare runtime state
        self._alloc_state()
        self.reset()

    def _alloc_state(self):
        B, N = self.B, self.N
        self.pos = torch.zeros((B, N, 2), dtype=torch.float32, device=self.device)
        self.vel = torch.zeros_like(self.pos)
        self.has_ball = torch.zeros((B, N), dtype=torch.bool, device=self.device)
        self.ball_pos = torch.zeros((B, 2), dtype=torch.float32, device=self.device)
        self.ball_vel = torch.zeros_like(self.ball_pos)
        self.score_blue = torch.zeros((B,), dtype=torch.int32, device=self.device)
        self.score_white = torch.zeros((B,), dtype=torch.int32, device=self.device)
        self.step_count = torch.zeros((B,), dtype=torch.int32, device=self.device)
        self.done = torch.zeros((B,), dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def reset(self):
        # Reset positions/velocities; place ball at center
        self.pos.copy_(self.initial_pos.unsqueeze(0).expand(self.B, -1, -1))
        self.vel.zero_()
        self.ball_pos.zero_()
        self.ball_vel.zero_()
        self.has_ball.zero_()
        self.score_blue.zero_()
        self.score_white.zero_()
        self.step_count.zero_()
        self.done.zero_()
        return self.compute_features()

    @torch.no_grad()
    def compute_features(self) -> torch.Tensor:
        """Return features [B,N,F] used by GA policy.
        Features (F=7): [bias, d_ball_bucket, d_goal_bucket, d_home_bucket, d_opp_bucket, teammate_open, goal_open]
        Buckets mirror the tabular discretization in qlearning.py.
        """
        B, N = self.B, self.N
        device = self.device

        # Distances
        # Ball: [B,N]
        d_ball = torch.linalg.norm(self.pos - self.ball_pos.unsqueeze(1), dim=-1)

        # Goal positions
        goal_left = torch.tensor([-self.width/2, 0.0], dtype=torch.float32, device=device)
        goal_right = torch.tensor([self.width/2, 0.0], dtype=torch.float32, device=device)

        # Distance to opponent goal and home goal depends on team
        # Build per-agent goal tensors [N,2]
        goal_for = torch.where(
            self.blue_mask.unsqueeze(-1), goal_right, goal_left
        ).to(torch.float32).to(device)
        goal_home = torch.where(
            self.blue_mask.unsqueeze(-1), goal_left, goal_right
        ).to(torch.float32).to(device)

        d_goal = torch.linalg.norm(self.pos - goal_for.unsqueeze(0), dim=-1)  # [B,N]
        d_home = torch.linalg.norm(self.pos - goal_home.unsqueeze(0), dim=-1)  # [B,N]

        # Distance to nearest opponent
        team_ids = torch.where(self.blue_mask, torch.zeros(self.N, dtype=torch.long, device=device), torch.ones(self.N, dtype=torch.long, device=device))
        # pairwise distances [B,N,N]
        diff = self.pos[:, :, None, :] - self.pos[:, None, :, :]
        dist = torch.linalg.norm(diff, dim=-1) + 1e-6
        # mask opponents
        opp_mask = (team_ids[None, :, None] != team_ids[None, None, :])
        masked_dist = torch.where(opp_mask, dist, torch.full_like(dist, 1e6))
        d_opp, _ = masked_dist.min(dim=-1)  # [B,N]

        # teammate_open: any teammate farther than threshold from all opponents (coarse)
        tm_mask = (team_ids[None, :, None] == team_ids[None, None, :]) & (~torch.eye(self.N, dtype=torch.bool, device=device)[None])
        tm_dists = torch.where(tm_mask, dist, torch.zeros_like(dist))
        # A teammate is "open" if there exists a teammate whose nearest opponent is farther than 10m
        opp_dists_vs_tm = torch.where(opp_mask, dist, torch.full_like(dist, 1e6))  # reuse dist
        nearest_opp_to_each_agent, _ = opp_dists_vs_tm.min(dim=-1)  # [B,N]
        # teammate_open per agent: does any teammate have nearest opponent > 10?
        teammate_open = (nearest_opp_to_each_agent > 10.0)
        # goal_open: nearest opponent to the goal center > 15m (proxy)
        goal_pos_per_agent = goal_for.unsqueeze(0).expand(B, -1, -1)
        opp_to_goal = torch.where(opp_mask, torch.linalg.norm(self.pos - goal_pos_per_agent, dim=-1), torch.full_like(dist, 1e6))
        nearest_opp_to_goal, _ = opp_to_goal.min(dim=-1)
        goal_open = nearest_opp_to_goal > 15.0

        # Discretize buckets
        d_ball_b = torch.clamp((d_ball / 10.0).floor(), 0, 4)
        d_goal_b = torch.clamp((d_goal / 20.0).floor(), 0, 4)
        d_home_b = torch.clamp((d_home / 20.0).floor(), 0, 4)
        d_opp_b = torch.clamp((d_opp / 10.0).floor(), 0, 4)

        features = torch.stack([
            torch.ones_like(d_ball_b),
            d_ball_b,
            d_goal_b,
            d_home_b,
            d_opp_b,
            teammate_open.to(d_ball_b.dtype),
            goal_open.to(d_ball_b.dtype),
        ], dim=-1)  # [B,N,7]
        return features

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform one step.
        actions: LongTensor [B,N] with indices into Actions list order.
        Returns: features [B,N,7], info dict with scores and done flags.
        """
        B, N = self.B, self.N
        device = self.device

        # -- Movement forces --
        force = torch.zeros_like(self.pos)

        # Precompute directions
        dir_to_ball = self.ball_pos.unsqueeze(1) - self.pos
        norm = torch.linalg.norm(dir_to_ball, dim=-1, keepdim=True) + 1e-6
        dir_to_ball = dir_to_ball / norm

        goal_left = torch.tensor([-self.width/2, 0.0], dtype=torch.float32, device=device)
        goal_right = torch.tensor([self.width/2, 0.0], dtype=torch.float32, device=device)
        goal_home = torch.where(self.blue_mask.unsqueeze(-1), goal_left, goal_right).to(device).to(torch.float32)

        # Action masks per index
        actions_list = list(Actions)
        idx_PASS = actions_list.index(Actions.PASS)
        idx_TACKLE = actions_list.index(Actions.TACKLE)
        idx_SHOOT = actions_list.index(Actions.SHOOT)
        idx_BLOCK = actions_list.index(Actions.BLOCK)
        idx_MOVE = actions_list.index(Actions.MOVE)
        idx_STAY = actions_list.index(Actions.STAY)

        is_MOVE = actions == idx_MOVE
        is_BLOCK = actions == idx_BLOCK
        is_STAY = actions == idx_STAY
        is_SHOOT = actions == idx_SHOOT
        is_PASS = actions == idx_PASS
        is_TACKLE = actions == idx_TACKLE

        # MOVE: towards ball
        force += is_MOVE.unsqueeze(-1) * (dir_to_ball * self.max_force)

        # BLOCK: towards home goal (defensive positioning)
        dir_to_home = goal_home.unsqueeze(0) - self.pos  # [B,N,2]
        dir_to_home = dir_to_home / (torch.linalg.norm(dir_to_home, dim=-1, keepdim=True) + 1e-6)
        force += is_BLOCK.unsqueeze(-1) * (dir_to_home * (self.max_force * 0.6))

        # STAY: damping only (handled below)

        # SHOOT/PASS effects when agent has ball
        has_ball_f = self.has_ball  # [B,N]
        # PASS: to nearest teammate
        if is_PASS.any():
            # Find nearest teammate per agent
            team_ids = torch.where(self.blue_mask, torch.zeros(self.N, dtype=torch.long, device=device), torch.ones(self.N, dtype=torch.long, device=device))
            diff = self.pos[:, :, None, :] - self.pos[:, None, :, :]
            dist = torch.linalg.norm(diff, dim=-1) + 1e-6
            tm_mask = (team_ids[None, :, None] == team_ids[None, None, :]) & (~torch.eye(self.N, dtype=torch.bool, device=device)[None])
            tm_dist = torch.where(tm_mask, dist, torch.full_like(dist, 1e6))
            nearest_tm_idx = tm_dist.argmin(dim=-1)  # [B,N]
            # For agents that have ball and chose PASS, set ball velocity toward nearest teammate
            choose = is_PASS & has_ball_f
            if choose.any():
                idx_b, idx_n = torch.where(choose)
                tm_idx = nearest_tm_idx[idx_b, idx_n]
                target = self.pos[idx_b, tm_idx, :]  # [K,2]
                origin = self.pos[idx_b, idx_n, :]
                v = target - origin
                v = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-6)
                self.ball_vel[idx_b] = v * 6.0
                self.has_ball[idx_b, idx_n] = False

        # SHOOT: toward opponent goal
        if is_SHOOT.any():
            goal_for = torch.where(self.blue_mask.unsqueeze(-1), goal_right, goal_left).to(device).to(torch.float32)
            choose = is_SHOOT & has_ball_f
            if choose.any():
                idx_b, idx_n = torch.where(choose)
                target = goal_for[idx_n, :]  # [K,2]
                origin = self.pos[idx_b, idx_n, :]
                v = target - origin
                v = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-6)
                self.ball_vel[idx_b] = v * 10.0
                self.has_ball[idx_b, idx_n] = False

        # TACKLE: move towards ball; if close to current ball, steal
        force += is_TACKLE.unsqueeze(-1) * (dir_to_ball * (self.max_force * 0.8))

        # Integrate agent motion: A1 = F*K1 - V*K2; V+=A; P+=V
        accel = force * self.K1 - self.vel * self.K2
        self.vel = self.vel + accel
        self.pos = self.pos + self.vel

        # Ball physics
        self.ball_pos = self.ball_pos + self.ball_vel
        self.ball_vel = self.ball_vel + (-self.ball_friction) * self.ball_vel

        # Possession update: if any agent close enough and no one holds ball, take it
        any_owner = self.has_ball.any(dim=1)  # [B]
        need_owner = ~any_owner
        if need_owner.any():
            idx_b = torch.where(need_owner)[0]
            # distances from these batch envs
            d = torch.linalg.norm(self.pos[idx_b] - self.ball_pos[idx_b].unsqueeze(1), dim=-1)  # [K,N]
            nearest = d.argmin(dim=-1)  # [K]
            close = d[torch.arange(idx_b.numel(), device=device), nearest] < self.possession_distance
            if close.any():
                idx_k = torch.where(close)[0]
                b_sel = idx_b[idx_k]
                n_sel = nearest[idx_k]
                self.has_ball[b_sel] = False
                self.has_ball[b_sel, n_sel] = True
                self.ball_vel[b_sel] = 0.0
                self.ball_pos[b_sel] = self.pos[b_sel, n_sel, :]

        # If an agent has ball, ball follows agent
        owners_b, owners_n = torch.where(self.has_ball)
        if owners_b.numel() > 0:
            self.ball_pos[owners_b] = self.pos[owners_b, owners_n, :]
            self.ball_vel[owners_b] = 0.0

        # TACKLE steal if close to ball owner
        if owners_b.numel() > 0:
            owner_pos = self.pos[owners_b, owners_n, :]
            d_owner = torch.linalg.norm(self.pos[owners_b] - owner_pos.unsqueeze(1), dim=-1)
            stealers = (d_owner < 2.5) & is_TACKLE[owners_b]
            if stealers.any():
                sb, sn = torch.where(stealers)
                real_b = owners_b[sb]
                # transfer ball to stealer
                self.has_ball[real_b] = False
                self.has_ball[real_b, sn] = True
                self.ball_pos[real_b] = self.pos[real_b, sn, :]
                self.ball_vel[real_b] = 0.0

        # Boundary: keep agents within field
        min_x, max_x = -self.width/2, self.width/2
        min_y, max_y = -self.height/2, self.height/2
        self.pos[..., 0].clamp_(min_x + 0.5, max_x - 0.5)
        self.pos[..., 1].clamp_(min_y + 0.5, max_y - 0.5)
        # Keep ball roughly in playable area unless shot/pass in progress (goals handled below)
        self.ball_pos[..., 0].clamp_(min_x - 1.0, max_x + 1.0)
        self.ball_pos[..., 1].clamp_(min_y - 1.0, max_y + 1.0)

        # Goal detection
        goal_y_half = self.goal_width / 2
        # Left goal (White scores)
        left_goal_mask = (self.ball_pos[..., 0] <= (min_x - 0.11)) & (self.ball_pos[..., 1].abs() <= goal_y_half)
        # Right goal (Blue scores)
        right_goal_mask = (self.ball_pos[..., 0] >= (max_x + 0.11)) & (self.ball_pos[..., 1].abs() <= goal_y_half)

        if left_goal_mask.any():
            idx = torch.where(left_goal_mask)[0]
            self.score_white[idx] += 1
            self.done[idx] = True
        if right_goal_mask.any():
            idx = torch.where(right_goal_mask)[0]
            self.score_blue[idx] += 1
            self.done[idx] = True

        # Step counter and episode done by time
        self.step_count += (~self.done).to(self.step_count.dtype)
        time_over = self.step_count >= self.max_steps
        self.done |= time_over

        features = self.compute_features()
        info = {
            'score_blue': self.score_blue.clone(),
            'score_white': self.score_white.clone(),
            'done': self.done.clone(),
        }
        return features, info
