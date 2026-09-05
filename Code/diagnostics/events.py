"""
diagnostics/events.py
======================
Eventos terminales y de detección para scipy.solve_ivp.
"""

from __future__ import annotations
import numpy as np


class HorizonCrossing:
    """Detecta R = rs. No terminal."""
    terminal  = False
    direction = -1

    def __init__(self, rs: float) -> None:
        self._rs = rs

    def __call__(self, tau: float, y: np.ndarray) -> float:
        return y[0] - self._rs


class SingularityApproach:
    """Termina cuando R < umbral."""
    terminal  = True
    direction = -1

    def __init__(self, R_threshold: float) -> None:
        self._R_th = R_threshold

    def __call__(self, tau: float, y: np.ndarray) -> float:
        return y[0] - self._R_th