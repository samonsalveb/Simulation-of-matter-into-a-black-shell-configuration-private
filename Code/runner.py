"""
runner.py — ShellRunner
========================
Orquesta la simulación completa a partir de ShellParams.

Uso mínimo:
    params = ShellParams(M=1.0, b=1.5).validate()
    result = ShellRunner(params).run()
    print(result.summary())

Benchmark analítico (solo a < 1 y a = 1):
    bench  = ShellRunner(params).get_analytic_benchmark()
    report = ShellRunner(params).validate(result, bench)
    print(report.summary())
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import warnings 

from .config import ShellParams
from .spacetimes.schwarzschild import Schwarzschild
from .spacetimes.minkowski import Minkowski
from .shell.dust import DustShell
from .shell.equations import ShellEOM
from .integrators.rk45_integrator import RK45Integrator, DOP853Integrator
from .integrators.rk4_integrator import RK4Integrator
from .integrators.base import IntegratorResult
from .diagnostics.monitor import ConservationMonitor

_INTEGRATORS = {
    "rk4":    RK4Integrator,
    "rk45":   RK45Integrator,
    "dop853": DOP853Integrator,
}


@dataclass
class ValidationReport:
    """
    Reporte de comparación entre trayectoria numérica y benchmark analítico.

    Campos
    ------
    max_R_error             : max |R_num(τ) − R_ana(τ)| sobre τ comunes
    mean_R_error            : media de |R_num − R_ana|
    max_C_numeric           : max |C| del integrador numérico
    max_C_analytic          : max |C| de la solución analítica (≈ ruido float64)
    C_overhead              : max_C_numeric / max_C_analytic
    epsilonR_drift_numeric  : drift de ε·R en la solución numérica
    epsilonR_drift_analytic : drift de ε·R en la solución analítica
    regime                  : régimen físico
    n_common_points         : puntos interpolados para la comparación
    passed                  : True si max_R_error < tol_R y max_C_numeric < tol_C
    tol_R                   : tolerancia usada para R
    tol_C                   : tolerancia usada para C
    """
    max_R_error:             float
    mean_R_error:            float
    max_C_numeric:           float
    max_C_analytic:          float
    C_overhead:              float
    epsilonR_drift_numeric:  float
    epsilonR_drift_analytic: float
    regime:                  str
    n_common_points:         int
    passed:                  bool
    tol_R:                   float
    tol_C:                   float

    def summary(self) -> str:
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return (
            f"ValidationReport [{self.regime}]  {status}\n"
            f"  Puntos comunes               : {self.n_common_points}\n"
            f"  max |R_num − R_ana|          : {self.max_R_error:.3e}  (tol={self.tol_R:.0e})\n"
            f"  mean|R_num − R_ana|          : {self.mean_R_error:.3e}\n"
            f"  max |C| numérico             : {self.max_C_numeric:.3e}  (tol={self.tol_C:.0e})\n"
            f"  max |C| analítico            : {self.max_C_analytic:.3e}\n"
            f"  C_overhead                   : {self.C_overhead:.1f}×\n"
            f"  ε·R drift numérico           : {self.epsilonR_drift_numeric:.3e}\n"
            f"  ε·R drift analítico          : {self.epsilonR_drift_analytic:.3e}\n"
        )


class ShellRunner:

    def __init__(self, params: ShellParams) -> None:
        self.params = params
        self._build()

    def _build(self) -> None:
        p = self.params

        self.exterior = Schwarzschild(M=p.M)
        self.interior = Minkowski()

        R0_ref      = p.R_max if np.isfinite(p.R_max) else 10.0 * p.rs
        self.matter = DustShell(M=p.M, R0=R0_ref)

        self.eom = ShellEOM(
            exterior=self.exterior,
            interior=self.interior,
            matter=self.matter,
            M=p.M,
            b=p.b,
        )

        self.integrator = _INTEGRATORS[p.integrator](eom=self.eom, params=p)
        self.monitor    = ConservationMonitor(eom=self.eom)
    
    
    def run(self) -> IntegratorResult:
        p = self.params

        # 1. Radio inicial
        if p.regime == "positive_binding":
            R0 = p.R_max
        else:
            R0 = 10.0 * p.rs

        # 2. Velocidad inicial
        if p.regime == "negative_binding":
            Rdot0 = np.sqrt(max((p.a + p.b / (2.0 * R0))**2 - 1.0, 0.0))
        else:
            Rdot0 = 0.0

        y0 = self.eom.initial_state(R0=R0, Rdot0=Rdot0)

        # 3. tau_span
        if p.a > 1.1:
            alpha     = np.arccosh(p.a)
            sech_a    = 1.0 / np.cosh(alpha)
            csch_a    = 1.0 / np.sinh(alpha)
            Theta_max = np.arcsinh(1.0 / np.cosh(alpha))
            tau_final = 1.5 * 0.5 * p.M * sech_a * csch_a**3 * (
                Theta_max * np.cosh(alpha) - np.sinh(Theta_max)
            )
        else:
            tau_final = 1.5 * p.tau_ff

        tau_span = (0.0, tau_final)

    # 4. Integración
        if p.regime == "negative_binding":
            from dataclasses import replace
            h_target = tau_final / p.n_points
            # max_step_fraction = h / tau_ff
            tau_ff_actual = np.pi * np.sqrt((10.0 * p.rs)**3 / (8.0 * p.M))
            msf_new = h_target / tau_ff_actual
            patched = replace(p, max_step_fraction=msf_new)
            from Code.integrators.rk4_integrator import RK4Integrator # type: ignore

            if p.integrator != "rk4":
                warnings.warn(
                    f"ShellRunner.run(): régimen negative_binding (a={p.a:.4f} > 1) "
                    f"fuerza RK4Integrator; se ignora params.integrator='{p.integrator}'.",
                    UserWarning,
                    stacklevel=2,
                )

            integrator_tmp = RK4Integrator(eom=self.eom, params=patched)
            return integrator_tmp.integrate(y0=y0, tau_span=tau_span)

        return self.integrator.integrate(y0=y0, tau_span=tau_span)

        # ── Benchmark analítico ───────────────────────────────────────────────────

    def get_analytic_benchmark(self) -> IntegratorResult:
        """
        Evalúa la solución analítica exacta de Israel (1967) Tabla I.

       Ahora soporta todos los regímenes

        Lanza
        -----
        NotImplementedError si a = M/b > 1 (régimen 'negative_binding').
        """
        from .benchmarks.analytic_solution import AnalyticSolution
        solver = AnalyticSolution(eom=self.eom, params=self.params)
        return solver.compute()

    def validate(
        self,
        numeric_result: IntegratorResult,
        benchmark: IntegratorResult | None = None,
        tol_R: float = 1e-6,
        tol_C: float = 1e-8,
        n_interp: int = 2000,
    ) -> ValidationReport:
        """
        Compara una trayectoria numérica contra el benchmark analítico.

        Interpola R_analítico sobre la grilla τ numérica, restringida
        al solapamiento [max(τ_min), min(τ_max)] de ambas trayectorias.

        Parámetros
        ----------
        numeric_result : IntegratorResult de ShellRunner.run()
        benchmark      : IntegratorResult de get_analytic_benchmark();
                         si None, se calcula automáticamente.
        tol_R          : tolerancia en error de R para passed=True
        tol_C          : tolerancia en max|C| numérico para passed=True
        n_interp       : puntos de interpolación

        Lanza
        -----
        NotImplementedError si el régimen es 'negative_binding'.
        ValueError si las trayectorias no se solapan en τ.
        """
        if benchmark is None:
            benchmark = self.get_analytic_benchmark()

        tau_lo = max(numeric_result.tau[0],  benchmark.tau[0])
        tau_hi = min(numeric_result.tau[-1], benchmark.tau[-1])

        if tau_lo >= tau_hi:
            raise ValueError(
                f"Trayectorias sin solapamiento en τ: "
                f"numérico [{numeric_result.tau[0]:.3f}, {numeric_result.tau[-1]:.3f}], "
                f"analítico [{benchmark.tau[0]:.3f}, {benchmark.tau[-1]:.3f}]."
            )

        tau_common   = np.linspace(tau_lo, tau_hi, n_interp)
        R_num_interp = np.interp(tau_common, numeric_result.tau, numeric_result.R)
        R_ana_interp = np.interp(tau_common, benchmark.tau,      benchmark.R)
        R_error      = np.abs(R_num_interp - R_ana_interp)

        # Usar max_C_exterior: excluye R < rs donde C explota por amplificación
        # numérica de b/2R, no por error del integrador.
        rs        = self.params.rs
        max_C_num = numeric_result.max_C_exterior(rs)
        max_C_ana = benchmark.max_C_exterior(rs)
        C_overhead = max_C_num / max_C_ana if max_C_ana > 0.0 else float("inf")

        eR_num             = numeric_result.epsilonR
        epsilonR_drift_num = float(np.max(np.abs(eR_num - eR_num[0])))

        passed = (float(np.max(R_error)) < tol_R) and (max_C_num < tol_C)

        return ValidationReport(
            max_R_error             = float(np.max(R_error)),
            mean_R_error            = float(np.mean(R_error)),
            max_C_numeric           = max_C_num,
            max_C_analytic          = max_C_ana,
            C_overhead              = C_overhead,
            epsilonR_drift_numeric  = epsilonR_drift_num,
            epsilonR_drift_analytic = benchmark.epsilonR_drift,
            regime                  = self.params.regime,
            n_common_points         = n_interp,
            passed                  = passed,
            tol_R                   = tol_R,
            tol_C                   = tol_C,
        )

    # ── Escaneo de parámetros ─────────────────────────────────────────────────

    def run_parameter_scan(self, param_name: str, values: list) -> list:
        from dataclasses import replace
        results = []
        for v in values:
            try:
                p_new = replace(self.params, **{param_name: v})
                p_new.validate()
                results.append(ShellRunner(p_new).run())
            except Exception as e:
                print(f"  [scan] {param_name}={v} → {e}")
                results.append(None)
        return results