"""
spacetimes/minkowski.py — Interior de Minkowski
================================================
Interior plano para la thin shell de polvo.

    f(R) = 1   (para todo R)
    f'(R) = 0

El interior Minkowski es la elección estándar para una shell
colapsando desde el reposo: no hay masa encerrada inicialmente.
Ref: Poisson §3.8.
"""

from __future__ import annotations
from .base import Spacetime


class Minkowski(Spacetime):
    """Spacetime de Minkowski (plano). Sin horizonte."""

    @property
    def rs(self) -> float:
        """Minkowski no tiene horizonte: rs = 0."""
        return 0.0

    def lapse(self, R: float) -> float:
        """f(R) = 1 para todo R."""
        return 1.0

    def dlapse_dR(self, R: float) -> float:
        """f'(R) = 0."""
        return 0.0

    def __repr__(self) -> str:
        return "Minkowski()"