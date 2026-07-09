"""
spacetimes/base.py — Motor Físico: Geometría
=============================================
Clase base abstracta para spacetimes del background.

Contrato de interfaz:
    Todo spacetime debe proveer f(R) y f'(R).
    La métrica exterior en coordenadas esféricas es:
        ds² = -f(R)dt² + f(R)⁻¹dR² + R²dΩ²

Nota sobre coordenadas:
    R es la coordenada radial de área (areal radius).
    En Schwarzschild, R coincide con r de Schwarzschild.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class Spacetime(ABC):
    """
    Interfaz para métricas esféricamente simétricas en forma estándar.

    La función de lapso f(R) determina completamente la geometría
    para el problema de thin shell (solo entra R en la EOM).
    """

    @abstractmethod
    def lapse(self, R: float) -> float:
        """
        Función de lapso f(R) tal que g_tt = -f(R).
        Ej: Schwarzschild → f(R) = 1 - 2M/R.
        """
        ...

    @abstractmethod
    def dlapse_dR(self, R: float) -> float:
        """
        Derivada f'(R) = df/dR. Necesaria para la aceleración d²R/dτ².
        """
        ...

    @property
    @abstractmethod
    def rs(self) -> float:
        """
        Radio característico (horizonte de eventos, si existe).
        Para Minkowski: rs = 0.
        """
        ...

    def is_outside_horizon(self, R: float) -> bool:
        """R > rs. Para Minkowski siempre True."""
        return R > self.rs