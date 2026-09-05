"""
shell/dust.py — Shell de polvo (presión cero)
==============================================
El modelo más simple: materia sin presión.

Conservación de masa superficial:
    σ(R) = σ₀ (R₀/R)²

Esta es consecuencia de ∇_a S^a_b = 0 con P=0:
    d(σ R²)/dτ = 0  →  σ R² = σ₀ R₀² = const.

Para la EOM de Israel con polvo, σ no aparece explícitamente
en la ecuación de R(τ) cuando se elimina usando la condición
de empalme (la masa M ya la encapsula). Sin embargo, σ es
necesaria para diagnósticos y para extensiones con presión.

Ref: Poisson §3.8, eq. (3.59)-(3.61).
"""

from __future__ import annotations
from .matter import MatterModel


class DustShell(MatterModel):
    """
    Shell de polvo: P = 0, σ ∝ R⁻².

    Parámetros
    ----------
    M : float
        Masa ADM. Para polvo, M = 4π σ₀ R₀² = masa total de la shell
        (en unidades geométricas).
        Nota: esta identificación asume interior Minkowski.
        Ref: Poisson §3.8, eq. (3.61).
    R0 : float
        Radio inicial.
    """

    def __init__(self, M: float, R0: float) -> None:
        self._M  = M
        self._R0 = R0
        # σ₀ = M / (4π R₀²)  [solo para diagnósticos, no entra en la EOM]
        import math
        self._sigma0 = M / (4.0 * math.pi * R0**2)

    @property
    def name(self) -> str:
        return "dust"

    def surface_density(self, R: float) -> float:
        """σ(R) = σ₀ (R₀/R)² = M/(4π R²)."""
        return self._sigma0 * (self._R0 / R)**2

    def pressure(self, R: float) -> float:
        """P = 0 para polvo."""
        return 0.0

    def __repr__(self) -> str:
        return f"DustShell(M={self._M}, R0={self._R0})"