"""
shell/matter.py — Modelos de materia para la thin shell
========================================================
La materia de la shell entra en la EOM a través de la densidad
superficial σ(R) y la presión superficial P(R).

Relación con las condiciones de Israel:
    Las ecuaciones de movimiento vienen de la condición de empalme:
        [K_ab] - h_ab [K] = -8π S_ab
    donde S_ab = diag(-σ, P, P) para una shell esféricamente simétrica.

    Para polvo: P = 0, y σ satisface la conservación ∇_a S^a_b = 0
    que da σ R² = const (conservación de masa superficial).

Ref: Israel (1966), eq. (3.7).
     Poisson, A Relativist's Toolkit, §3.7-3.8.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class MatterModel(ABC):
    """
    Modelo de materia de la thin shell.

    Provee σ(R) y P(R) en función del radio actual.
    Internamente puede guardar condiciones iniciales para
    satisfacer conservación de energía-momento.
    """

    @abstractmethod
    def surface_density(self, R: float) -> float:
        """
        Densidad superficial σ(R) [masa/área].
        Siempre >= 0 (condición de energía débil).
        """
        ...

    @abstractmethod
    def pressure(self, R: float) -> float:
        """
        Presión superficial P(R).
        Para polvo: P = 0.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador del modelo para logging."""
        ...