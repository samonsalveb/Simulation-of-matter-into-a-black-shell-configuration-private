"""
diagnostics/monitor.py
========================
Monitor de conservación durante la integración.

Monitorea C (polvo, actual) y tiene acceso a ε (universal, fase 2).

C = 1 + Ṙ² - (a + b/2R)²  = 0     ← monitoreo activo
ε = √(f₊+Ṙ²) - √(f₋+Ṙ²) = -b/R   ← disponible para diagnóstico

Para polvo: |C| < 1e-8 es aceptable con RK45.
            |C| < 1e-10 con DOP853.
Cuando se agregue presión: cambiar a monitorear ε directamente.
"""

from __future__ import annotations
import warnings
import numpy as np
from typing import Optional


class ConservationMonitor:
    """
    Monitor de la cantidad conservada C durante la integración.

    Parámetros
    ----------
    eom : ShellEOM
        Provee C(y) y epsilon(R, Rdot).
    warn_threshold : float
        |C| que dispara advertencia. Default: 1e-8.
    stop_threshold : float | None
        |C| que aborta la integración. Default: 1e-4.
    """

    def __init__(
        self,
        eom,
        warn_threshold:  float = 1e-8,
        stop_threshold:  Optional[float] = 1e-4,
    ) -> None:
        self._eom           = eom
        self._warn_th       = warn_threshold
        self._stop_th       = stop_threshold
        self._warned        = False
        self.max_C_seen     = 0.0
        self.n_checks       = 0

        # Historial de ε·R para diagnóstico de presión (fase 2)
        self._epsilonR_history: list[float] = []

    def check(self, tau: float, y: np.ndarray) -> None:
        """
        Evalúa |C| en el estado actual.
        También registra ε·R para diagnóstico futuro.
        """
        R, Rdot = y[0], y[1]

        # ── Monitoreo activo: C ───────────────────────────────────────────────
        C_now = abs(self._eom.C(y))
        self.max_C_seen = max(self.max_C_seen, C_now)
        self.n_checks  += 1

        if C_now > self._warn_th and not self._warned:
            warnings.warn(
                f"ConservationMonitor: |C| = {C_now:.2e} en τ={tau:.4f}. "
                f"Considera DOP853 o reducir max_step_fraction.",
                RuntimeWarning, stacklevel=2,
            )
            self._warned = True

        if self._stop_th is not None and C_now > self._stop_th:
            raise RuntimeError(
                f"ConservationMonitor: |C| = {C_now:.2e} supera "
                f"stop_threshold={self._stop_th:.2e}. Integración abortada."
            )

        # ── Diagnóstico pasivo: ε·R ───────────────────────────────────────────
        # Para polvo: ε·R = -b = const.
        # Si deriva, indica presión emergente.
        eps_R = self._eom.epsilon(R, Rdot) * R
        self._epsilonR_history.append(eps_R)

    @property
    def epsilonR_drift(self) -> Optional[float]:
        """
        Drift de ε·R respecto al valor inicial.
        Para polvo puro debe ser ≈ 0.
        Útil en fase 2 para detectar efectos de presión.
        """
        if len(self._epsilonR_history) < 2:
            return None
        ref = self._epsilonR_history[0]
        return float(max(abs(v - ref) for v in self._epsilonR_history))

    def summary(self) -> str:
        eps_drift = self.epsilonR_drift
        eps_str   = f"{eps_drift:.2e}" if eps_drift is not None else "n/a"
        return (
            f"ConservationMonitor:\n"
            f"  max |C|        = {self.max_C_seen:.2e}   (polvo, activo)\n"
            f"  ε·R drift      = {eps_str}   (universal, diagnóstico)\n"
            f"  n_checks       = {self.n_checks}\n"
            f"  warn_threshold = {self._warn_th:.2e}\n"
        )