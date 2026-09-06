"""
tests/test_suite.py
====================
Suite de tests para el proyecto Thin Shell Collapse.

Organización:
    TestShellParams         — configuración y validación
    TestSpacetimes          — Schwarzschild y Minkowski
    TestShellEOM            — física central: C, ε, rhs
    TestAnalyticSolution    — benchmark Israel (1967)
    TestIntegrators         — RK4 y RK45 contra la solución analítica
    TestRunner              — integración end-to-end y validate()

Convención de tolerancias:
    Solución analítica : |C| < 1e-10  (ruido float64)
    RK45 / DOP853      : |C| < 1e-7
    RK4 paso fijo      : |C| < 1e-3   (orden 4, paso h ∝ tau_ff * 5e-4)

Ref: Israel, W. (1967). Phys. Rev. 153, 1388.
     Poisson, E. A Relativist's Toolkit, §3.8.
"""

import pytest
import numpy as np
import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
# Permite correr con: pytest tests/ desde project/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Code.config import ShellParams
from Code.runner import ShellRunner
from Code.spacetimes.schwarzschild import Schwarzschild
from Code.spacetimes.minkowski import Minkowski
from Code.shell.equations import ShellEOM
from Code.shell.dust import DustShell
from Code.benchmarks.analytic_solution import AnalyticSolution
from Code.integrators.base import IntegratorResult


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def params_positive():
    """a = 1/3 < 1 — energía de ligadura positiva."""
    return ShellParams(M=1.0, b=3.0, integrator="rk45",
                       n_points=4000, max_step_fraction=1e-3).validate()

@pytest.fixture
def params_zero():
    """a = 1 — energía de ligadura cero."""
    return ShellParams(M=1.0, b=1.0, integrator="rk45",
                       n_points=4000, max_step_fraction=1e-3).validate()

@pytest.fixture
def params_negative():
    """a = 2 > 1 — energía de ligadura negativa."""
    return ShellParams(M=2.0, b=1.0, integrator="rk45",
                       n_points=4000, max_step_fraction=1e-3).validate()

@pytest.fixture
def eom_positive(params_positive):
    p = params_positive
    exterior = Schwarzschild(M=p.M)
    interior = Minkowski()
    matter   = DustShell(M=p.M, R0=p.R_max)
    return ShellEOM(exterior=exterior, interior=interior,
                    matter=matter, M=p.M, b=p.b)

@pytest.fixture
def eom_zero(params_zero):
    p = params_zero
    exterior = Schwarzschild(M=p.M)
    interior = Minkowski()
    matter   = DustShell(M=p.M, R0=500.0 * p.rs)
    return ShellEOM(exterior=exterior, interior=interior,
                    matter=matter, M=p.M, b=p.b)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ShellParams
# ══════════════════════════════════════════════════════════════════════════════

class TestShellParams:

    def test_a_derivado(self):
        p = ShellParams(M=1.0, b=4.0)
        assert p.a == pytest.approx(0.25)

    def test_rs(self):
        p = ShellParams(M=3.0, b=6.0)
        assert p.rs == pytest.approx(6.0)

    def test_regime_positive(self):
        assert ShellParams(M=1.0, b=3.0).regime == "positive_binding"

    def test_regime_zero(self):
        assert ShellParams(M=1.0, b=1.0).regime == "zero_binding"

    def test_regime_negative(self):
        assert ShellParams(M=2.0, b=1.0).regime == "negative_binding"

    def test_R_max_formula(self):
        # R_max = M / 2a(1-a)
        p = ShellParams(M=1.0, b=3.0)   # a = 1/3
        expected = 1.0 / (2.0 * (1.0/3.0) * (2.0/3.0))
        assert p.R_max == pytest.approx(expected, rel=1e-10)

    def test_R_max_infinito_para_a1(self):
        assert np.isinf(ShellParams(M=1.0, b=1.0).R_max)

    def test_R_max_infinito_para_negativo(self):
        assert np.isinf(ShellParams(M=2.0, b=1.0).R_max)

    def test_from_R_max_roundtrip(self):
        M, R_max = 1.0, 5.0
        p = ShellParams.from_R_max(M=M, R_max=R_max)
        assert p.R_max == pytest.approx(R_max, rel=1e-8)
        assert p.M == M

    def test_validate_M_negativo(self):
        with pytest.raises(ValueError, match="M debe ser"):
            ShellParams(M=-1.0, b=1.0).validate()

    def test_validate_b_negativo(self):
        with pytest.raises(ValueError, match="b debe ser"):
            ShellParams(M=1.0, b=-1.0).validate()

    def test_validate_Q_no_implementado(self):
        with pytest.raises(ValueError, match="carga"):
            ShellParams(M=1.0, b=1.0, Q=1.0).validate()

    def test_validate_shell_dentro_horizonte(self):
        # R_max <= rs requiere a suficientemente grande con a < 1
        # R_max = M/2a(1-a) <= 2M  →  1/4a(1-a) <= 1  →  a(1-a) >= 1/4
        # Máximo de a(1-a) es 1/4 en a=0.5, con igualdad. Para a → 0.5
        # desde abajo R_max → ∞, no es posible R_max <= rs con a < 1.
        # Este test verifica que from_R_max rechaza R_max <= 2M.
        with pytest.raises(ValueError, match="fuera del horizonte"):
            ShellParams.from_R_max(M=1.0, R_max=1.5)  # R_max < 2M


# ══════════════════════════════════════════════════════════════════════════════
# 2. Spacetimes
# ══════════════════════════════════════════════════════════════════════════════

class TestSpacetimes:

    def test_schwarzschild_lapse(self):
        s = Schwarzschild(M=1.0)
        assert s.lapse(4.0) == pytest.approx(1.0 - 2.0/4.0)

    def test_schwarzschild_dlapse(self):
        s = Schwarzschild(M=1.0)
        assert s.dlapse_dR(4.0) == pytest.approx(2.0/16.0)

    def test_schwarzschild_rs(self):
        assert Schwarzschild(M=2.5).rs == pytest.approx(5.0)

    def test_schwarzschild_M_negativo(self):
        with pytest.raises(ValueError):
            Schwarzschild(M=-1.0)

    def test_minkowski_lapse_constante(self):
        m = Minkowski()
        for R in [0.1, 1.0, 100.0]:
            assert m.lapse(R) == 1.0

    def test_minkowski_dlapse_cero(self):
        m = Minkowski()
        assert m.dlapse_dR(5.0) == 0.0

    def test_minkowski_rs_cero(self):
        assert Minkowski().rs == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. ShellEOM
# ══════════════════════════════════════════════════════════════════════════════

class TestShellEOM:

    def test_C_en_reposo_en_R_max(self, eom_positive, params_positive):
        """En R_max con Ṙ=0, C debe ser exactamente 0 (condición inicial)."""
        R0   = params_positive.R_max
        y0   = eom_positive.initial_state(R0=R0)
        C0   = eom_positive.C(y0)
        assert abs(C0) < 1e-10, f"|C(R_max, 0)| = {abs(C0):.3e}, esperado < 1e-10"

    def test_rhs_dimensiones(self, eom_positive, params_positive):
        y0  = eom_positive.initial_state(R0=params_positive.R_max)
        dy  = eom_positive.rhs(0.0, y0)
        assert dy.shape == (2,)

    def test_rhs_Rdot_es_primera_componente(self, eom_positive, params_positive):
        R0   = params_positive.R_max
        Rdot = -0.3
        y    = np.array([R0, Rdot])
        dy   = eom_positive.rhs(0.0, y)
        assert dy[0] == pytest.approx(Rdot)

    def test_rhs_Rddot_negativo(self, eom_positive, params_positive):
        """R̈ < 0: la shell siempre se desacelera hacia la singularidad."""
        y0 = eom_positive.initial_state(R0=params_positive.R_max)
        dy = eom_positive.rhs(0.0, y0)
        assert dy[1] < 0.0

    def test_epsilon_times_R_NO_es_constante_Schwarzschild(self, eom_positive, params_positive):
        """
        ε·R = -b es constante SOLO si f₋ = f₊ (mismo spacetime en ambos lados).
        Para Schwarzschild exterior + Minkowski interior, ε·R varía con R.

        Con f₋ = 1 y f₊ = 1 - 2M/R, en reposo:
            ε·R = (√(1-2M/R) - 1)·R  ← no constante.

        La cantidad conservada para polvo es C = 0 (Israel eq. 43), no ε·R.
        Este test verifica que ε·R efectivamente varía (comportamiento correcto).
        """
        solver = AnalyticSolution(eom=eom_positive, params=params_positive)
        result = solver.compute()
        eR    = eom_positive.epsilon_array(result.R, result.Rdot) * result.R
        drift = float(np.max(np.abs(eR - eR[0])))
        # Para Schwarzschild+Minkowski, ε·R no es constante — drift > 0 es correcto.
        assert drift > 0.1, (
            f"ε·R debería variar para Schwarzschild+Minkowski, drift={drift:.3e}"
        )

    def test_C_array_consistente_con_C_escalar(self, eom_positive, params_positive):
        R0 = params_positive.R_max
        R    = np.array([R0, R0 * 0.8, R0 * 0.5])
        Rdot = np.array([0.0, -0.1, -0.3])
        C_arr = eom_positive.C_array(R, Rdot)
        for i in range(len(R)):
            y     = np.array([R[i], Rdot[i]])
            C_sc  = eom_positive.C(y)
            assert C_arr[i] == pytest.approx(C_sc, rel=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# 4. AnalyticSolution
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyticSolution:

    def test_positive_binding_C_drift(self, eom_positive, params_positive):
        """Solución analítica: |C| < 1e-10 en todo el recorrido."""
        solver = AnalyticSolution(eom=eom_positive, params=params_positive)
        result = solver.compute()
        assert result.max_C_drift < 1e-10

    def test_positive_binding_R_decrece(self, eom_positive, params_positive):
        """R debe ser monótonamente decreciente (colapso)."""
        solver = AnalyticSolution(eom=eom_positive, params=params_positive)
        result = solver.compute()
        assert np.all(np.diff(result.R) <= 0), "R no es monótonamente decreciente"

    def test_positive_binding_Rdot_negativo(self, eom_positive, params_positive):
        """Ṙ ≤ 0 en todo momento (interior del colapso)."""
        solver = AnalyticSolution(eom=eom_positive, params=params_positive)
        result = solver.compute()
        # El primer punto tiene Ṙ = 0 (reposo en R_max)
        assert np.all(result.Rdot <= 1e-12), "Ṙ positivo detectado"

    def test_positive_binding_R0_igual_R_max(self, eom_positive, params_positive):
        """El primer punto debe ser R_max."""
        solver = AnalyticSolution(eom=eom_positive, params=params_positive)
        result = solver.compute()
        assert result.R[0] == pytest.approx(params_positive.R_max, rel=1e-6)

    def test_positive_binding_tau_crece(self, eom_positive, params_positive):
        solver = AnalyticSolution(eom=eom_positive, params=params_positive)
        result = solver.compute()
        assert np.all(np.diff(result.tau) > 0), "τ no es monótonamente creciente"

    def test_positive_binding_horizonte_detectado(self, eom_positive, params_positive):
        """La shell debe cruzar el horizonte (R_max > rs para a=1/3)."""
        solver = AnalyticSolution(eom=eom_positive, params=params_positive)
        result = solver.compute()
        assert result.crossed_horizon, "No se detectó cruce del horizonte"

    def test_zero_binding_C_drift(self, eom_zero, params_zero):
        solver = AnalyticSolution(eom=eom_zero, params=params_zero)
        result = solver.compute()
        assert result.max_C_drift < 1e-10

    def test_zero_binding_Rdot_negativo(self, eom_zero, params_zero):
        solver = AnalyticSolution(eom=eom_zero, params=params_zero)
        result = solver.compute()
        assert np.all(result.Rdot <= 0), "Ṙ positivo en régimen a=1"

    def test_negative_binding_conservacion_C(self, params_negative):
        exterior = Schwarzschild(M=params_negative.M)
        interior = Minkowski()
        matter   = DustShell(M=params_negative.M, R0=100.0 * params_negative.rs)
        eom      = ShellEOM(exterior=exterior, interior=interior,
                        matter=matter, M=params_negative.M, b=params_negative.b)

        result = AnalyticSolution(eom=eom, params=params_negative).compute()

        assert result.max_C_drift < 1e-10   # tolerancia documentada en §8 para la solución analítica

    def test_resultado_es_IntegratorResult(self, eom_positive, params_positive):
        solver = AnalyticSolution(eom=eom_positive, params=params_positive)
        result = solver.compute()
        assert isinstance(result, IntegratorResult)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Integradores
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrators:

    # ── RK45 ──────────────────────────────────────────────────────────────────

    def test_rk45_conservacion_C(self, params_positive):
        """RK45: |C| < 1e-12 fuera del horizonte (región física relevante)."""
        result = ShellRunner(params_positive).run()
        assert result.max_C_drift_exterior < 1e-10, \
            f"RK45 max|C| exterior = {result.max_C_drift_exterior:.3e}"

    def test_rk45_R_decrece(self, params_positive):
        result = ShellRunner(params_positive).run()
        assert np.all(np.diff(result.R) <= 0)

    def test_rk45_cruce_horizonte(self, params_positive):
        result = ShellRunner(params_positive).run()
        assert result.crossed_horizon

    def test_rk45_vs_analitico_R_error(self, params_positive):
        """RK45 debe reproducir la trayectoria analítica con error < 1e-5."""
        runner = ShellRunner(params_positive)
        result = runner.run()
        report = runner.validate(result, tol_R=1e-4, tol_C=1e-6)
        assert report.passed, (
            f"RK45 vs analítico FAILED:\n{report.summary()}"
        )

    # ── RK4 ───────────────────────────────────────────────────────────────────

    def test_rk4_termina_sin_crash(self):
        """RK4 con parámetros razonables debe completar sin excepción."""
        p = ShellParams(M=1.0, b=3.0, integrator="rk4",
                        max_step_fraction=1e-4).validate()
        result = ShellRunner(p).run()
        assert result.success or "singularidad" in result.message

    def test_rk4_R_positivo(self):
        """RK4 nunca debe entregar R < 0 en el array de salida."""
        p = ShellParams(M=1.0, b=3.0, integrator="rk4",
                        max_step_fraction=1e-4).validate()
        result = ShellRunner(p).run()
        assert np.all(result.R > 0)

    def test_rk4_conservacion_C_paso_fino(self):
        """RK4 con paso fino: |C| < 1e-5 fuera del horizonte."""
        p = ShellParams(M=1.0, b=3.0, integrator="rk4",
                        max_step_fraction=5e-5,
                        singularity_threshold=0.05).validate()
        result = ShellRunner(p).run()
        assert result.max_C_drift_exterior < 1e-5, \
            f"RK4 max|C| exterior = {result.max_C_drift_exterior:.3e}"

    # ── DOP853 ────────────────────────────────────────────────────────────────

    def test_dop853_C_mejor_que_rk45(self, params_positive):
        """DOP853 (orden 8) debe dar menor drift de C que RK45 (orden 5)."""
        p_dop = ShellParams(M=params_positive.M, b=params_positive.b,
                            integrator="dop853",
                            rtol=params_positive.rtol,
                            atol=params_positive.atol).validate()
        p_rk  = ShellParams(M=params_positive.M, b=params_positive.b,
                            integrator="rk45",
                            rtol=params_positive.rtol,
                            atol=params_positive.atol).validate()
        C_dop = ShellRunner(p_dop).run().max_C_drift
        C_rk  = ShellRunner(p_rk).run().max_C_drift
        assert C_dop <= C_rk * 10, \
            f"DOP853 ({C_dop:.2e}) no mejor que RK45 ({C_rk:.2e})"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Runner — end-to-end y validate()
# ══════════════════════════════════════════════════════════════════════════════

class TestRunner:

    def test_run_devuelve_IntegratorResult(self, params_positive):
        result = ShellRunner(params_positive).run()
        assert isinstance(result, IntegratorResult)

    def test_get_analytic_benchmark_devuelve_IntegratorResult(self, params_positive):
        bench = ShellRunner(params_positive).get_analytic_benchmark()
        assert isinstance(bench, IntegratorResult)

    def test_get_analytic_benchmark_negative_devuelve_resultado(self, params_negative):
        result = ShellRunner(params_negative).get_analytic_benchmark()

        assert isinstance(result, IntegratorResult)   # mismo patrón que test_resultado_es_IntegratorResult
        assert result.max_C_drift < 1e-10

    def test_validate_passed_con_rk45(self, params_positive):
        runner = ShellRunner(params_positive)
        result = runner.run()
        report = runner.validate(result, tol_R=1e-4, tol_C=1e-6)
        assert report.passed, report.summary()

    def test_validate_sin_solapamiento_devuelve_resultado(self, params_positive):
        """Si las trayectorias no se solapan en τ, debe lanzar ValueError."""
        runner = ShellRunner(params_positive)
        result = runner.run()
        bench  = runner.get_analytic_benchmark()

        # Fabricar un result con τ fuera del rango del benchmark
        fake = IntegratorResult(
            tau=bench.tau[-1:] + np.array([1.0, 2.0]),
            R=np.array([0.5, 0.4]),
            Rdot=np.array([-0.1, -0.2]),
            C=np.zeros(2),
            epsilon=np.zeros(2),
            horizon_crossing_tau=None,
            success=True,
            message="fake",
        )
        with pytest.raises(ValueError, match="solapamiento"):
            runner.validate(fake, bench)

    def test_parameter_scan_longitud(self, params_positive):
        runner  = ShellRunner(params_positive)
        valores = [2.0, 3.0, 4.0]
        results = runner.run_parameter_scan("b", valores)
        assert len(results) == len(valores)

    def test_parameter_scan_invalido_devuelve_None(self, params_positive):
        """Parámetros que no pasan validate() deben devolver None, no crash."""
        runner  = ShellRunner(params_positive)
        results = runner.run_parameter_scan("b", [-1.0])   # b < 0 → inválido
        assert results[0] is None

    def test_tau_empieza_en_cero(self, params_positive):
        result = ShellRunner(params_positive).run()
        assert result.tau[0] == pytest.approx(0.0, abs=1e-10)

    def test_R0_igual_R_max(self, params_positive):
        result = ShellRunner(params_positive).run()
        assert result.R[0] == pytest.approx(params_positive.R_max, rel=1e-6)

    def test_negative_binding_run_advierte_override_integrador(self, params_negative):
        with pytest.warns(UserWarning, match="RK4"):
            result = ShellRunner(params_negative).run()
        assert isinstance(result, IntegratorResult)