"""
integrators/base.py
====================
Contrato base. IntegratorResult reporta:
    C       — conservación de energía (polvo): debe ser ~0
    epsilon — cantidad universal: para polvo ε·R = -b = const
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class IntegratorResult:
    tau:                  np.ndarray
    R:                    np.ndarray
    Rdot:                 np.ndarray
    C:                    np.ndarray
    epsilon:              np.ndarray
    horizon_crossing_tau: Optional[float]
    success:              bool
    message:              str
    rs:                   Optional[float] = None   # radio de Schwarzschild = 2M
    params:               dict = field(default_factory=dict)

    @property
    def max_C_drift(self) -> float:
        """max|C| sobre toda la trayectoria, incluyendo R → 0."""
        return float(np.max(np.abs(self.C)))

    @property
    def max_C_drift_exterior(self) -> float:
        """
        max|C| restringido a R > rs (exterior al horizonte).

        Cerca de la singularidad Ṙ → -∞ y pequeños errores numéricos
        en R producen C grande por amplificación en b/2R. Ese ruido
        no refleja la calidad del integrador en la región física relevante.

        Lanza ValueError si rs no fue proporcionado al construir el resultado.
        """
        if self.rs is None:
            raise ValueError(
                "rs no fue proporcionado en este IntegratorResult. "
                "Usa max_C_exterior(rs) con rs explícito."
            )
        mask = self.R > self.rs
        if not np.any(mask):
            return float(np.max(np.abs(self.C)))
        return float(np.max(np.abs(self.C[mask])))

    def max_C_exterior(self, rs: float) -> float:
        """
        max|C| restringido a R > rs. Versión con rs explícito,
        útil cuando el resultado fue construido sin rs.
        """
        mask = self.R > rs
        if not np.any(mask):
            return float(np.max(np.abs(self.C)))
        return float(np.max(np.abs(self.C[mask])))

    @property
    def epsilonR(self) -> np.ndarray:
        return self.epsilon * self.R

    @property
    def epsilonR_drift(self) -> float:
        eR = self.epsilonR
        return float(np.max(np.abs(eR - eR[0])))

    @property
    def crossed_horizon(self) -> bool:
        return self.horizon_crossing_tau is not None

    def summary(self) -> str:
        hc = (f"{self.horizon_crossing_tau:.4f}"
              if self.crossed_horizon else "no ocurrió")
        rs_str = f"{self.rs:.4f}" if self.rs is not None else "no proporcionado"
        ext_str = (f"{self.max_C_drift_exterior:.2e}"
                   if self.rs is not None else "n/a (rs no proporcionado)")
        return (
            f"IntegratorResult:\n"
            f"  success             = {self.success}\n"
            f"  message             = {self.message}\n"
            f"  n_steps             = {len(self.tau)}\n"
            f"  tau_total           = {self.tau[-1]:.4f}\n"
            f"  R[0]               = {self.R[0]:.4f}\n"
            f"  R_min              = {np.min(self.R):.6f}\n"
            f"  rs                 = {rs_str}\n"
            f"  horizon_crossing   = τ = {hc}\n"
            f"  max |C| global     = {self.max_C_drift:.2e}   (toda trayectoria)\n"
            f"  max |C| exterior   = {ext_str}   (R > rs)\n"
            f"  ε·R drift          = {self.epsilonR_drift:.2e}\n"
        )


class Integrator(ABC):

    def __init__(self, eom, params, monitor=None) -> None:
        self.eom     = eom
        self.params  = params
        self.monitor = monitor

    @abstractmethod
    def integrate(self, y0: np.ndarray, tau_span: tuple) -> IntegratorResult:
        ...

    def _compute_C(self, R: np.ndarray, Rdot: np.ndarray) -> np.ndarray:
        return self.eom.C_array(R, Rdot)

    def _compute_epsilon(self, R: np.ndarray, Rdot: np.ndarray) -> np.ndarray:
        return self.eom.epsilon_array(R, Rdot)

    def _find_horizon_crossing(self, tau, R, rs) -> Optional[float]:
        idx = np.where(R < rs)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        if i == 0:
            return float(tau[0])
        frac = (rs - R[i-1]) / (R[i] - R[i-1])
        return float(tau[i-1] + frac * (tau[i] - tau[i-1]))