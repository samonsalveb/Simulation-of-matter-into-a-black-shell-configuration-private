"""
config.py — Cuarto de Control
==============================
Parámetros fundamentales: (M, b, Q, J)

Notación (consistente con Israel 1967 y Poisson):
    M  : masa gravitacional total
    b  : masa nucleónica (masa en reposo de los constituyentes)
    a  : M/b — controla energía de ligadura (derivado)
    Q  : carga eléctrica exterior (Reissner-Nordström) — fase 2
    J  : momento angular (Kerr) — fase 3

Tres regímenes (Israel 1967, Tabla I):
    a < 1  →  energía de ligadura positiva, R_max = M/2a(1-a)
    a = 1  →  energía de ligadura cero, R_max = ∞
    a > 1  →  energía de ligadura negativa

Cantidad conservada universal (sin carga):
    ε = √(f₊(R) + Ṙ²) - √(f₋(R) + Ṙ²) = const
    donde f₊, f₋ son los lapsos exterior e interior.
    Ver shell/equations.py.

Con carga (pendiente de referencia):
    ε incluye términos q²/2R y qQ/R.
    Q_ext = Q_int + q (conservación de carga).
    No implementado hasta tener referencia clara.

Unidades geométricas: G = c = 1.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

# "exact" eliminado: la solución analítica no es un integrador.
# Acceder a ella via ShellRunner.get_analytic_benchmark().
IntegratorName = Literal["rk4", "rk45", "dop853"]
MatterModel    = Literal["dust"]


@dataclass
class ShellParams:
    # ── Parámetros fundamentales ──────────────────────────────────────────────
    M:  float   # masa gravitacional
    b:  float   # masa nucleónica

    # Fase 2 — carga (no implementado, reservado)
    Q:  float = 0.0

    # Fase 3 — momento angular (no implementado, reservado)
    J:  float = 0.0

    # ── Control numérico ──────────────────────────────────────────────────────
    matter:                MatterModel    = "dust"
    integrator:            IntegratorName = "rk4"
    rtol:                  float          = 1e-10
    atol:                  float          = 1e-12
    max_step_fraction:     float          = 5e-4
    n_points:              int            = 4000
    singularity_threshold: float          = 0.01

    # ── Propiedades derivadas ─────────────────────────────────────────────────

    @property
    def a(self) -> float:
        return self.M / self.b

    @property
    def rs(self) -> float:
        return 2.0 * self.M

    @property
    def regime(self) -> str:
        a = self.a
        if a < 1.0 - 1e-10:
            return "positive_binding"
        elif abs(a - 1.0) < 1e-10:
            return "zero_binding"
        else:
            return "negative_binding"

    @property
    def R_max(self) -> float:
        """R_max = M / 2a(1-a). Solo válido para a < 1."""
        a = self.a
        if a >= 1.0:
            return np.inf
        return self.M / (2.0 * a * (1.0 - a))

    @property
    def alpha(self) -> float:
        a = self.a
        if a < 1.0:
            return float(np.arccos(a))
        elif a > 1.0:
            return float(np.arccosh(a))
        return 0.0

    @property
    def tau_ff(self) -> float:
        R_ref = self.R_max if self.regime == "positive_binding" else 10.0 * self.rs
        return np.pi * np.sqrt(R_ref**3 / (8.0 * self.M))

    @property
    def epsilon_initial(self) -> Optional[float]:
        if self.Q != 0.0 or self.J != 0.0:
            return None
        return None

    # ── Constructor alternativo ───────────────────────────────────────────────

    @classmethod
    def from_R_max(cls, M: float, R_max: float, **kwargs) -> "ShellParams":
        if R_max <= 2.0 * M:
            raise ValueError(
                f"R_max={R_max} debe ser > 2M={2*M} (fuera del horizonte)."
            )
        discriminant = 1.0 - (2*M) / R_max
        if discriminant < 0:
            raise ValueError(f"R_max={R_max} demasiado pequeño para M={M}.")
        a = (1.0 - np.sqrt(discriminant)) / 2.0
        b = M / a
        return cls(M=M, b=b, **kwargs)

    # ── Validación ────────────────────────────────────────────────────────────

    def validate(self) -> "ShellParams":
        errors = []

        if self.M <= 0:
            errors.append(f"M debe ser > 0, recibido M={self.M}")
        if self.b <= 0:
            errors.append(f"b debe ser > 0, recibido b={self.b}")
        if self.Q != 0.0:
            errors.append(
                "Q ≠ 0: carga eléctrica no implementada aún. Usar Q=0.0."
            )
        if self.J != 0.0:
            errors.append("J ≠ 0: momento angular no implementado aún.")
        if self.regime == "positive_binding" and self.R_max <= self.rs:
            errors.append(
                f"R_max={self.R_max:.4f} <= rs={self.rs:.4f}. "
                f"Shell empieza dentro del horizonte."
            )
        if self.rtol < 1e-15 or self.rtol > 1e-3:
            errors.append(f"rtol={self.rtol} fuera de [1e-15, 1e-3].")
        if self.singularity_threshold < 1e-4:
            errors.append("singularity_threshold mínimo recomendado: 1e-4.")

        if errors:
            msg = "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
            raise ValueError(f"ShellParams inválidos:\n{msg}")
        return self

    def summary(self) -> str:
        rmax = f"{self.R_max:.4f}" if self.regime == "positive_binding" else "∞"
        return (
            f"ShellParams:\n"
            f"  M          = {self.M}\n"
            f"  b          = {self.b}\n"
            f"  a = M/b    = {self.a:.6f}  [{self.regime}]\n"
            f"  rs         = {self.rs:.4f}\n"
            f"  R_max      = {rmax}\n"
            f"  α          = {self.alpha:.4f} rad\n"
            f"  Q          = {self.Q}  (carga, no implementada)\n"
            f"  J          = {self.J}  (momento angular, no implementado)\n"
            f"  integrator = {self.integrator}\n"
            f"  rtol/atol  = {self.rtol} / {self.atol}\n"
        )