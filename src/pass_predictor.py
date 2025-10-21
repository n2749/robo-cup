from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


# Lookup tables to keep implementation lightweight while exposing the core idea
PASSER_ROLE_WEIGHTS: Dict[str, float] = {
    "attacker": 0.08,
    "midfielder": 0.10,
    "defender": -0.05,
    "goalkeeper": -0.08,
}

TARGET_ROLE_WEIGHTS: Dict[str, float] = {
    "attacker": 0.05,
    "midfielder": 0.08,
    "defender": -0.02,
    "goalkeeper": -0.04,
}

ROLE_SPEED_SCALE: Dict[str, float] = {
    "attacker": 1.05,
    "midfielder": 1.0,
    "defender": 0.95,
    "goalkeeper": 0.85,
}

ROLE_PRESSURE_TOLERANCE: Dict[str, float] = {
    "attacker": 1.1,
    "midfielder": 1.0,
    "defender": 0.9,
    "goalkeeper": 0.85,
}


def _role_multiplier(role: str, table: Dict[str, float]) -> float:
    return table.get(role, 0.0)


def _confidence_bucket(p_skill: float, r_skill: float, attempts: int, calibration_error: float) -> float:
    """Return a numeric confidence score in [0, 1]."""
    def _norm_skill(skill: float) -> float:
        return (max(0.6, min(1.4, skill)) - 0.6) / (1.4 - 0.6)
    
    skill_factor = (_norm_skill(p_skill) + _norm_skill(r_skill)) / 2.0
    attempts_factor = min(attempts / 20.0, 1.0)
    calibration_factor = max(0.0, min(1.0, 1.0 - calibration_error))
    return 0.35 * skill_factor + 0.35 * attempts_factor + 0.30 * calibration_factor


def _confidence_label(score: float) -> str:
    if score < 0.35:
        return "Low"
    if score < 0.7:
        return "Medium"
    return "High"


def predict_pass_success(passer, receiver, env) -> Dict[str, Optional[float]]:
    """
    Lightweight, role-aware heuristic that mimics the requested Bayesian upgrade.
    Returns probability estimate alongside a confidence score and the features
    that influenced the calculation.
    """
    passer_role = getattr(passer, "role", "")
    receiver_role = getattr(receiver, "role", "")

    # Distances & kinematics
    vector = receiver.pos - passer.pos
    distance = float(np.linalg.norm(vector))
    angle = math.degrees(math.atan2(vector[1], vector[0])) if distance > 1e-5 else 0.0

    passer_speed = float(np.linalg.norm(passer.vel)) * ROLE_SPEED_SCALE.get(passer_role, 1.0)
    target_speed = float(np.linalg.norm(receiver.vel)) * ROLE_SPEED_SCALE.get(receiver_role, 1.0)

    # Pressure approximated via distance to nearest opponent
    opponents = [a for a in env.agents if a.team != passer.team]
    if opponents:
        pressure = min(float(np.linalg.norm(passer.pos - opp.pos)) for opp in opponents)
        defender_proximity = min(float(np.linalg.norm(receiver.pos - opp.pos)) for opp in opponents)
    else:
        pressure = 999.0
        defender_proximity = 999.0

    # Pass type heuristic (ground / lofted / long)
    if distance < 10.0:
        pass_type = "short"
    elif distance < 25.0:
        pass_type = "medium"
    else:
        pass_type = "long"

    # Base probability starts moderate
    probability = 0.55

    # Distance dampening
    probability -= min(distance / 60.0, 0.25)

    # Pressure adjustments (closer opponents reduce odds)
    pressure_penalty = 0.0 if pressure > 15.0 else (15.0 - pressure) / 50.0
    probability -= pressure_penalty

    # Defender proximity to target
    tolerance = ROLE_PRESSURE_TOLERANCE.get(receiver_role, 1.0)
    defender_threshold = 10.0 * tolerance
    defender_penalty = 0.0 if defender_proximity > defender_threshold else (defender_threshold - defender_proximity) / 40.0
    probability -= defender_penalty

    # Role multipliers
    probability += _role_multiplier(passer_role, PASSER_ROLE_WEIGHTS)
    probability += _role_multiplier(receiver_role, TARGET_ROLE_WEIGHTS)

    # Movement adjustments
    probability += min(target_speed / 12.0, 0.08)  # moving target helps a little
    probability -= min(passer_speed / 15.0, 0.05)  # hard passes on the run slightly riskier

    # Skill scaling (acts like player-specific accuracy)
    skill = getattr(passer, "pass_skill", 1.0)
    receiver_skill = getattr(receiver, "receive_skill", 1.0)
    combined_skill = max(0.5, min(1.4, (skill * 0.6 + receiver_skill * 0.4)))
    probability *= combined_skill

    # Calibration adjustment based on recent prediction error feedback
    calibration_error = getattr(env, "pass_calibration_error", 0.5)
    probability *= 0.8 + 0.4 * max(0.0, min(1.0, 1.0 - calibration_error))

    # Safety clamp
    probability = max(0.05, min(0.95, probability))

    attempts = getattr(passer, "pass_attempts", 0)
    confidence_score = _confidence_bucket(skill, receiver_skill, attempts, calibration_error)
    confidence_label = _confidence_label(confidence_score)

    return {
        "probability": probability,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "features": {
            "distance": distance,
            "angle": angle,
            "pressure": pressure,
            "defender_proximity": defender_proximity,
            "pass_type": pass_type,
            "passer_speed": passer_speed,
            "target_speed": target_speed,
            "passer_role": passer_role,
            "target_role": receiver_role,
            "pass_skill": skill,
            "receiver_skill": receiver_skill,
            "attempts": attempts,
            "calibration_error": calibration_error,
        },
    }
