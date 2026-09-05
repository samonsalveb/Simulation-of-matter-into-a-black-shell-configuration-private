"""
integrators/rk45_integrator.py — RK45 y DOP853
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from .base import Integrator, IntegratorResult


class RK45Integrator(Integrator):
    METHOD = "RK45"

    def integrate(self, y0: np.ndarray, tau_span: tuple) -> IntegratorResult:
        rs       = self.params.rs
        R_stop   = self.params.singularity_threshold * rs
        max_step = self.params.tau_ff * self.params.max_step_fraction

        def singularity(tau, y):
            return y[0] - R_stop
        singularity.terminal  = True
        singularity.direction = -1

        sol = solve_ivp(
            fun=self.eom.rhs,
            t_span=tau_span,
            y0=y0,
            method=self.METHOD,
            events=[singularity],
            max_step=max_step,
            rtol=self.params.rtol,
            atol=self.params.atol,
        )

        tau     = sol.t
        R       = sol.y[0]
        Rdot    = sol.y[1]
        C       = self._compute_C(R, Rdot)
        epsilon = self._compute_epsilon(R, Rdot)

        return IntegratorResult(
            tau=tau, R=R, Rdot=Rdot,
            C=C, epsilon=epsilon,
            horizon_crossing_tau=self._find_horizon_crossing(tau, R, rs),
            success=sol.success,
            message=sol.message,
            rs=rs,
            params={"method": self.METHOD, "n_eval": sol.nfev,
                    "M": self.params.M, "b": self.params.b},
        )


class DOP853Integrator(RK45Integrator):
    METHOD = "DOP853"