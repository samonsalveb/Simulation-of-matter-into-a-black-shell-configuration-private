"""
spacetimes/schwarzschild.py — Exterior de Schwarzschild
========================================================
Métrica exterior para el colapso de thin shell.

    f(R) = 1 - 2M/R
    f'(R) = 2M/R²

Ref: Poisson, A Relativist's Toolkit, eq. (3.1).
     Israel, Il Nuovo Cimento B, 44(1), 1966.
"""

from __future__ import annotations
from .base import Spacetime


class Schwarzschild(Spacetime):
    """
    Spacetime de Schwarzschild con masa ADM M.

    Parámetros
    ----------
    M : float
        Masa en unidades geométricas (G=c=1). M > 0.
    """

    def __init__(self, M: float) -> None:
        if M <= 0:
            raise ValueError(f"Schwarzschild requiere M > 0, recibido M={M}")
        self._M = M

    @property
    def M(self) -> float:
        return self._M

    @property
    def rs(self) -> float:
        """Radio de Schwarzschild r_s = 2M."""
        return 2.0 * self._M

    def lapse(self, R: float) -> float:
        """f(R) = 1 - 2M/R."""
        return 1.0 - 2.0 * self._M / R

    def dlapse_dR(self, R: float) -> float:
        """f'(R) = 2M/R²."""
        return 2.0 * self._M / (R * R)

    def __repr__(self) -> str:
        return f"Schwarzschild(M={self._M})"