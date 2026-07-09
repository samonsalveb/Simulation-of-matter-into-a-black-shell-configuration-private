"""
benchmarks/analytic_solution.py
================================
Soluciones paramétricas exactas de Israel (1967), Tabla I.

Caso 1 — a < 1: IMPLEMENTADO y verificado.
Caso 2 — a = 1: IMPLEMENTADO y verificado.
Caso 3 — a > 1: IMPLIMENTADO y verificado.
caso 4 — a → ∞: NO IMPLEMENTADO (límite singular, no es un shell de polvo).

NO hereda de Integrator. No recibe ODE. El método público es compute(),
no integrate(y0, tau_span). Devuelve IntegratorResult para compatibilidad
con ShellRunner.validate().

Ref: Israel, W. (1967). Phys. Rev. 153, 1388. Tabla I.
"""

from __future__ import annotations
import numpy as np
from ..integrators.base import IntegratorResult


class AnalyticSolution:
    """
    Evaluador de soluciones analíticas exactas para shell esférica de polvo.

    Parámetros
    ----------
    eom    : ShellEOM  — para calcular C y ε sobre la trayectoria.
    params : ShellParams
    """

    def __init__(self, eom, params) -> None:
        self.eom    = eom
        self.params = params

        #if params.a > 1.0 + 1e-10:
        #    raise NotImplementedError(
        #        f"No existe solución analítica cerrada para a = M/b = {params.a:.6f} > 1 "
        #        f"(régimen 'negative_binding'). Use integrator='rk45' o 'dop853'."
        #    )

    def compute(self) -> IntegratorResult:
        a = self.params.a
        if a < 1.0 - 1e-10:
            return self._positive_binding(a)
        elif abs(a - 1) < 1e-10:
            return self._zero_binding()
        elif a <= 100.0:
            return self._negative_binding(a)
        else:
            return self._null_shell_limit(a)

    # ── régimen a < 1 ────────────────────────────────────────────────────────

    def _positive_binding(self, a: float) -> IntegratorResult:
        """
        Israel (1967) Tabla I — energía de ligadura positiva (a < 1).

            R(Θ) = ½m · secα · csc²α · (cosα − cosΘ)
            τ(Θ) = ½m · secα · csc³α · (Θ·cosα − sinΘ) + const

        donde α = arccos(a),  Θ ∈ [π, 2π − α).

        Derivadas:
            dR/dΘ   = ½m·secα·csc²α · sinΘ           < 0  en (π, 2π−α)
            dτ/dΘ   = ½m·secα·csc³α · (cosα − cosΘ)  > 0  en (π, 2π−α)
            Ṙ = dR/dΘ / dτ/dΘ < 0  ✓ (colapso, sin corrección de signo)
        """
        m      = self.params.M
        rs     = self.params.rs
        n      = self.params.n_points
        R_stop = self.params.singularity_threshold * rs

        alpha = np.arccos(a)
        sec_a = 1.0 / np.cos(alpha)
        csc_a = 1.0 / np.sin(alpha)

        Theta = np.linspace(np.pi, 2.0*np.pi - alpha - 1e-6, n)

        R   = 0.5 * m * sec_a * csc_a**2 * (np.cos(alpha) - np.cos(Theta))
        tau = 0.5 * m * sec_a * csc_a**3 * (Theta * np.cos(alpha) - np.sin(Theta))
        tau = tau - tau[0]

        mask    = R > R_stop
        if not np.any(mask):
            mask[0] = True
        R       = R[mask]
        tau     = tau[mask]
        Theta_m = Theta[mask]

        dR_dTheta   = 0.5 * m * sec_a * csc_a**2 * np.sin(Theta_m)
        dtau_dTheta = 0.5 * m * sec_a * csc_a**3 * (np.cos(alpha) - np.cos(Theta_m))

        with np.errstate(invalid='ignore', divide='ignore'):
            Rdot = np.where(
                np.abs(dtau_dTheta) > 1e-15,
                dR_dTheta / dtau_dTheta,
                0.0
            )

        C         = self.eom.C_array(R, Rdot)
        epsilon   = self.eom.epsilon_array(R, Rdot)
        tau_cross = self._find_horizon_crossing(tau, R, rs)

        return IntegratorResult(
            tau=tau, R=R, Rdot=Rdot,
            C=C, epsilon=epsilon,
            horizon_crossing_tau=tau_cross,
            success=True,
            message=f"Exacta Israel (1967) Tabla I, a<1 (α={alpha:.4f} rad).",
            rs=rs,
            params={
                "method": "analytic_positive_binding",
                "a": a, "b": self.params.b, "M": m,
            },
        )

    # ── régimen a = 1 ────────────────────────────────────────────────────────

    def _zero_binding(self) -> IntegratorResult:
        """
        Israel (1967) Tabla I — energía de ligadura cero (a = 1).

            R(λ) = m(λ² − ¼)
            τ(λ) = m(⅔λ³ − ½λ) + const

        λ decrece desde λ_i (R = 500·rs) hasta ½ + ε (R → 0).

        Derivadas (dλ < 0):
            dR/dλ   =  2mλ           > 0
            dτ/dλ   = −m(2λ² − ½)   < 0  para λ > ½
            Ṙ = dR/dλ / dτ/dλ < 0  ✓ (colapso, sin corrección de signo)
        """
        m      = self.params.M
        rs     = self.params.rs
        n      = self.params.n_points
        R_stop = self.params.singularity_threshold * rs
        R0     = 500.0 * rs

        lambda_i = np.sqrt(R0 / m + 0.25)
        lambda_f = 0.5 + 1e-8

        lam = np.linspace(lambda_i, lambda_f, n)

        R       = m * (lam**2 - 0.25)
        tau_raw = m * (2.0/3.0 * lam**3 - 0.5 * lam)
        tau     = tau_raw[0] - tau_raw

        mask = R > R_stop
        if not np.any(mask):
            mask[0] = True
        R    = R[mask]
        tau  = tau[mask]
        lam  = lam[mask]

        dR_dlam   =  2.0 * m * lam
        dtau_dlam = -m * (2.0 * lam**2 - 0.5)

        with np.errstate(invalid='ignore', divide='ignore'):
            Rdot = np.where(
                np.abs(dtau_dlam) > 1e-15,
                dR_dlam / dtau_dlam,
                0.0
            )

        C         = self.eom.C_array(R, Rdot)
        epsilon   = self.eom.epsilon_array(R, Rdot)
        tau_cross = self._find_horizon_crossing(tau, R, rs)

        return IntegratorResult(
            tau=tau, R=R, Rdot=Rdot,
            C=C, epsilon=epsilon,
            horizon_crossing_tau=tau_cross,
            success=True,
            message="Exacta Israel (1967) Tabla I, a=1 (energía de ligadura cero).",
            rs=rs,
            params={
                "method": "analytic_zero_binding",
                "a": 1.0, "b": self.params.b, "M": m,
            },
        )

    # ── régimen a > 1 ─────────────────────────────────────────────────────
    
    def _negative_binding(self, a: float) -> IntegratorResult:
        """
        Israel (1967) Tabla I — energía de ligadura negativa (a > 1).
            
            R(Θ) = ½m · sechα · csch²α · (coshα − coshΘ)
            τ(Θ) = ½m · sechα · csch³α · (Θcoshα − sinhΘ)+ const

        donde α = arccosh(a),  Θ ∈ [0, 2π).

        Derivadas:
            dR/dΘ   = ½m·sechα·csch²α · sinhΘ          < 0  en (0, π)
            dτ/dΘ   = ½m·sechα·csch³α · (sinhα·coshα − Θ·coshα + sinΘ) < 0 en (0, π)
            Ṙ = dR/dΘ / dτ/dΘ < 0  ✓ (colapso, sin corrección de signo)
        """
        m = self.params.M
        rs = self.params.rs
        n = self.params.n_points
        R_stop = self.params.singularity_threshold * rs
        R0 = 500.0 * rs # el coefiente es muy plano. Se puede ajustar para mejorar la resolución cerca del horizonte, pero no es crítico.
        # R0 se elige grande para que el shell comience lejos del horizonte, donde la solución es más estable numéricamente. El rango de Θ se ajusta dinámicamente para cubrir desde R0 hasta el horizonte (R=0).
        # y mirar el integrador. Si el integrador es estable, se puede reducir R0 para mejorar la resolución cerca del horizonte.

        alpha = np.arccosh(a)
        sech_a = 1.0 / np.cosh(alpha)
        csch_a = 1.0 / np.sinh(alpha)
        prefactor = 0.5 * m * sech_a * csch_a**2

        csch_Theta_start = R0 / prefactor + np.cosh(alpha)
        Theta_start = np.arcsinh(1.0 / csch_Theta_start)

        Theta_max = np.arcsinh(1.0 / np.cosh(alpha))

        
        # Θ decrece de Theta_start a alpha+ε  (R decrece de R0 a 0)
        Theta = np.linspace(Theta_start, Theta_max - 1e-8, n)
        
        csch_Theta = 1.0 / np.sinh(Theta)

        R  = prefactor * (csch_Theta - np.cosh(alpha))
        tau = 0.5 * m * sech_a * csch_a**3 * (Theta * np.cosh(alpha) - np.sinh(Theta))
        tau = tau - tau[0]

        
        mask = R > R_stop
        if not np.any(mask):
            mask[0] = True 
        R = R[mask]
        tau = tau[mask] 
        Theta_m = Theta[mask]

        csch_Theta_m = 1.0 / np.sinh(Theta_m)
        coth_Theta_m = np.cosh(Theta_m) / np.sinh(Theta_m)  

        dR_dTheta = -prefactor * csch_Theta_m * coth_Theta_m
        dtau_dTheta = 0.5 * m * sech_a * csch_a**3 * (np.cosh(alpha) - np.cosh(Theta_m))    

        val  = (a + self.params.b / (2.0 * R))**2 - 1.0
        Rdot = -np.sqrt(np.maximum(val, 0.0))
        
        C = self.eom.C_array(R, Rdot)
        epsilon = self.eom.epsilon_array(R, Rdot)
        tau_cross = self._find_horizon_crossing(tau, R, rs)

        return IntegratorResult(
            tau=tau, R=R, Rdot=Rdot,
            C=C, epsilon=epsilon,
            horizon_crossing_tau=tau_cross,
            success=True,
            message = f"Exacta Israel (1967) Tabla I, a>1 (α={alpha:.4f} rad).",
            rs=rs,
            params={
                "method": "analytic_negative_binding",
                "a": a, "b": self.params.b, "M": m,
            },
        )

    # ── régimen a  → ∞  ──────────────────────────────────────────────────────    

    def _null_shell_limit(self, a: float) -> IntegratorResult:
        """
        Límite singular a → ∞ (b → 0). No es un shell de polvo físico.
        """
        m = self.params.M
        rs = self.params.rs
        n = self.params.n_points
        R_stop = self.params.singularity_threshold * rs
        R0 = 500.0 * rs

        # se usa un pseudo-tiempo
        R  = np.linspace(R0, rs + 1e-7, n)

        t_coords = -(R + rs * np.log(np.abs(R/rs - 1.0) + 1e-15))
        tau_axis = t_coords - t_coords[0]

        Rdot = -np.ones_like(R)

        return IntegratorResult(
            tau=tau_axis, 
            R=R, 
            Rdot=Rdot,
            C=np.zeros_like(R), 
            epsilon=np.zeros_like(R),
            horizon_crossing_tau=None,
            success=True,
            message=f"Límite singular a → ∞ (b → 0). No es un shell de polvo físico.",
            rs=rs,
            params={
                "method": "analytic_null_shell_limit",
                "a": a, "b": self.params.b, "M": m,
            },
        )


    # ── utilidad interna ─────────────────────────────────────────────────────

    @staticmethod
    def _find_horizon_crossing(
        tau: np.ndarray, R: np.ndarray, rs: float
    ) -> float | None:
        idx = np.where(R < rs)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        if i == 0:
            return float(tau[0])
        frac = (rs - R[i-1]) / (R[i] - R[i-1])
        return float(tau[i-1] + frac * (tau[i] - tau[i-1]))