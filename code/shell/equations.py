"""
shell/equations.py — EOM de thin shell
========================================
Dos cantidades relevantes:

1. Conservación de energía — C (Israel 1967, eq. 10):
   C = 1 + Ṙ² - (a + b/2R)² = 0
   Específica para polvo. Origen: M = b√(1+Ṙ²) - b²/2R (Poisson §3.8).
   Esta es la cantidad que se monitorea en la integración actual.

2. Cantidad universal — ε:
   ε = √(f₊(R) + Ṙ²) - √(f₋(R) + Ṙ²)
   Para polvo: ε = -b/R (no constante, pero ε·R = -b = const).
   Para shell con presión: ε = -4πRσ = función de la materia.
   Útil cuando se agregue presión — el monitor tendrá acceso a ε
   para detectar cuándo deja de valer la aproximación de polvo.

EOM de segundo orden (derivando C=0 respecto a τ, cancelando Ṙ≠0):
   R̈ = -(b/2R²)(a + b/2R)

Estado integrado: y = [R, Ṙ] — Ṙ es variable de estado, no se reconstruye.

Ref: Israel (1966), Il Nuovo Cimento B 44(1).
     Israel (1967), Phys. Rev. 153, 1388.
     Poisson, A Relativist's Toolkit, §3.8.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..spacetimes.base import Spacetime
    from .matter import MatterModel


class ShellEOM:
    """
    EOM para thin shell de polvo.

    Parámetros
    ----------
    exterior : Spacetime   f₊(R), f₊'(R)
    interior : Spacetime   f₋(R), f₋'(R)
    matter   : MatterModel
    M : float   masa gravitacional
    b : float   masa nucleónica
    """

    def __init__(
        self,
        exterior: "Spacetime",
        interior: "Spacetime",
        matter:   "MatterModel",
        M:        float,
        b:        float,
    ) -> None:
        self.exterior = exterior
        self.interior = interior
        self.matter   = matter
        self._M       = M
        self._b       = b
        self._a       = M / b

    # ── EOM ──────────────────────────────────────────────────────────────────

    def rhs(self, tau: float, y: np.ndarray) -> np.ndarray:
        """
        dy/dτ = [Ṙ, R̈]
        R̈ = -(b/2R²)(a + b/2R)
        Derivado de C=0 respecto a τ. Ref: Israel (1967) eq.(10).
        """
        R    = y[0]
        Rdot = y[1]
        b    = self._b
        a    = self._a
        Rddot = -(b / (2.0 * R**2)) * (a + b / (2.0 * R))
        return np.array([Rdot, Rddot])

    # ── Cantidad conservada de energía (polvo) ────────────────────────────────

    def C(self, y: np.ndarray) -> float:
        """
        C = 1 + Ṙ² - (a + b/2R)² = 0

        Conservación de energía para thin shell de polvo.
        Origen: M = b√(1+Ṙ²) - b²/2R  (Poisson §3.8).
        Ref: Israel (1967), eq. (10).

        Monitorear |C| durante la integración — debe ser < 1e-8.
        """
        R, Rdot = y[0], y[1]
        return 1.0 + Rdot**2 - (self._a + self._b / (2.0 * R))**2

    def C_array(self, R: np.ndarray, Rdot: np.ndarray) -> np.ndarray:
        """Versión vectorizada de C para arrays."""
        return 1.0 + Rdot**2 - (self._a + self._b / (2.0 * R))**2

    # ── Cantidad universal ε ──────────────────────────────────────────────────

    def epsilon(self, R: float, Rdot: float) -> float:
        """
        ε = √(f₊ + Ṙ²) - √(f₋ + Ṙ²)

        Cantidad universal — no específica de polvo.
        Para polvo: ε = -b/R  (no constante, pero ε·R = -b = const).
        Para shell con presión: ε = -4πRσ.

        Útil para:
        - Detectar cuándo la aproximación de polvo falla (presión emergente).
        - Extender a Reissner-Nordström cambiando solo f₊.
        - Monitorear en régimen de presión (fase 2).
        """
        fp = self.exterior.lapse(R)
        fm = self.interior.lapse(R)
        return np.sqrt(max(fp + Rdot**2, 0.0)) - np.sqrt(max(fm + Rdot**2, 0.0))

    def epsilon_array(self, R: np.ndarray, Rdot: np.ndarray) -> np.ndarray:
        """Versión vectorizada de epsilon para arrays."""
        fp = np.array([self.exterior.lapse(r) for r in R])
        fm = np.array([self.interior.lapse(r) for r in R])
        return np.sqrt(np.maximum(fp + Rdot**2, 0.0)) - np.sqrt(np.maximum(fm + Rdot**2, 0.0))

    def epsilon_times_R(self, R: np.ndarray, Rdot: np.ndarray) -> np.ndarray:
        """
        ε·R = -b para polvo (constante).
        Si ε·R deriva de -b, indica que la materia ya no es polvo puro.
        Diagnóstico para extensión con presión.
        """
        return self.epsilon_array(R, Rdot) * R

    # ── Estado inicial ────────────────────────────────────────────────────────

    def initial_state(self, R0: float, Rdot0: float = 0.0) -> np.ndarray:
        """
        y₀ = [R₀, -|Ṙ₀|]. Ṙ < 0 para colapso.
        Para shell en reposo: Ṙ₀ = 0.
        """
        return np.array([R0, -abs(Rdot0)])

    def __repr__(self) -> str:
        return (
            f"ShellEOM(M={self._M}, b={self._b}, a={self._a:.6f}, "
            f"exterior={self.exterior!r})"
        )