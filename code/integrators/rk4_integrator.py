"""
integrators/rk4_integrator.py — RK4 paso fijo
================================================
Implementación propia de Runge-Kutta orden 4 con paso uniforme.

Algoritmo:
    k1 = f(τₙ,       yₙ)
    k2 = f(τₙ + h/2, yₙ + h/2·k1)
    k3 = f(τₙ + h/2, yₙ + h/2·k2)
    k4 = f(τₙ + h,   yₙ + h·k3)
    yₙ₊₁ = yₙ + h/6·(k1 + 2k2 + 2k3 + k4)

donde f(τ, y) = eom.rhs(τ, y),  y = [R, Ṙ].

Paso: h = tau_ff * max_step_fraction  (uniforme, fijo)

Paradas:
    R < singularity_threshold * rs  →  termina integración
    R < rs                          →  registra cruce del horizonte
"""

from __future__ import annotations
import numpy as np
from .base import Integrator, IntegratorResult


class RK4Integrator(Integrator):

    def integrate(self, y0: np.ndarray, tau_span: tuple) -> IntegratorResult:
        tau_start, tau_end = tau_span

        rs     = self.params.rs
        R_stop = self.params.singularity_threshold * rs
        h      = self.params.tau_ff * self.params.max_step_fraction

        tau_list  = [tau_start]
        R_list    = [y0[0]]
        Rdot_list = [y0[1]]

        tau = tau_start
        y   = y0.copy()

        horizon_crossing_tau = None
        success = True
        message = "Integración completada."

        while tau < tau_end:
            if tau + h > tau_end:
                h = tau_end - tau

            k1 = self.eom.rhs(tau,       y)
            k2 = self.eom.rhs(tau + h/2, y + h/2 * k1)
            k3 = self.eom.rhs(tau + h/2, y + h/2 * k2)
            k4 = self.eom.rhs(tau + h,   y + h   * k3)

            y_next = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            tau   += h

            if horizon_crossing_tau is None and y_next[0] < rs <= y[0]:
                frac = (rs - y[0]) / (y_next[0] - y[0])
                horizon_crossing_tau = (tau - h) + frac * h

            y = y_next
            R = y[0]

            if R <= 0.0 or R < R_stop:
                message = (
                    f"Detenido antes de la singularidad "
                    f"(R={R:.4f} < R_stop={R_stop:.4f})."
                )
                break

            tau_list.append(tau)
            R_list.append(y[0])
            Rdot_list.append(y[1])

        tau_arr  = np.array(tau_list)
        R_arr    = np.array(R_list)
        Rdot_arr = np.array(Rdot_list)

        C       = self._compute_C(R_arr, Rdot_arr)
        epsilon = self._compute_epsilon(R_arr, Rdot_arr)

        return IntegratorResult(
            tau=tau_arr,
            R=R_arr,
            Rdot=Rdot_arr,
            C=C,
            epsilon=epsilon,
            horizon_crossing_tau=horizon_crossing_tau,
            success=success,
            message=message,
            rs=rs,
            params={
                "method": "RK4",
                "h": h,
                "n_steps": len(tau_arr),
                "max_step_fraction": self.params.max_step_fraction,
                "M": self.params.M,
                "b": self.params.b,
            },
        )