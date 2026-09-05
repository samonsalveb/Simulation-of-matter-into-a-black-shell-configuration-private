"""
Mapa de condiciones iniciales — Colapso de cáscara delgada de polvo
====================================================================
Retrato de fase (R, Ṙ) + potencial efectivo V(R), para una familia del
parámetro de ligadura a = M/b. Muestra TODA la familia de soluciones (no
una trayectoria): puntos de retorno, horizonte, los tres regímenes de
Israel, y el locus de "black shell" en el límite de polvo (a = 1/2).

Diseño (decisiones deliberadas, ver conversación):
  · El núcleo es la ECUACIÓN MAESTRA DE EMPALME en términos de f±(R), NO la
    forma horneada (a + b/2R)². Eso hace el mapa charge-ready: el día que
    metas Reissner–Nordström, solo cambia f_exterior; nada más se toca.
  · La forma cerrada (a + b/2R)² queda como BENCHMARK, validado contra el
    núcleo en _selftest() (mismo principio que ExactParametric: benchmark,
    no integrador).

Ecuación maestra (Israel 1966/1967; Poisson, A Relativist's Toolkit §3.8):
    √(Ṙ² + f₋) − √(Ṙ² + f₊) = m/R ,     m = b  (masa en reposo, polvo)

Autor del andamiaje: para la tesis de Santiago Monsalve (OAN, UNAL).
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Interfaz de spacetime  (hook charge-ready)
# Cuando integres al paquete, reemplazá estas dos por
#   Schwarzschild(M).lapse(R)  y  Minkowski().lapse(R)
# La firma es la misma f(R): no cambia nada más.
# ─────────────────────────────────────────────────────────────────────────────
def f_exterior(R, M, Q=0.0):
    """f₊(R). Q=0 → Schwarzschild.  Q≠0 → Reissner–Nordström (charge-ready)."""
    return 1.0 - 2.0 * M / R + Q**2 / R**2

def f_interior(R):
    """f₋(R) = 1  (Minkowski)."""
    return np.ones_like(np.asarray(R, dtype=float))

# ─────────────────────────────────────────────────────────────────────────────
# Núcleo: ÚNICA fuente de verdad para la cinemática del shell
# ─────────────────────────────────────────────────────────────────────────────
def rdot2(R, M, b, Q=0.0):
    """
    Ṙ²(R) despejado de la ecuación maestra de empalme.
        β₋ − β₊ = m/R ,   β₊² − β₋² = f₊ − f₋   (m = b)
      ⇒ β₋ = ½[ b/R + R(f₋ − f₊)/b ] ,   Ṙ² = β₋² − f₋
    """
    R  = np.asarray(R, dtype=float)
    fm = f_interior(R)
    fp = f_exterior(R, M, Q)
    beta_minus = 0.5 * (b / R + R * (fm - fp) / b)
    return beta_minus**2 - fm

def V_eff(R, M, b, Q=0.0):
    """Potencial efectivo: Ṙ² = −V.  Permitido V≤0, retorno V=0, prohibido V>0."""
    return -rdot2(R, M, b, Q)

def turning_point(a, b):
    """R de retorno (Ṙ=0). Solo existe en régimen ligado a<1 (polvo, Q=0)."""
    return b / (2.0 * (1.0 - a)) if a < 1.0 else None

def regime(a):
    if a < 1.0:  return "ligado (a<1)"
    if a == 1.0: return "marginal (a=1)"
    return "no ligado (a>1)"

# ─────────────────────────────────────────────────────────────────────────────
# Validación: la forma cerrada Schwarzschild+Minkowski debe salir del núcleo
# ─────────────────────────────────────────────────────────────────────────────
def _selftest():
    M, b = 1.0, 1.7
    a = M / b
    R = np.linspace(2.1, 20, 60)
    closed = (a + b / (2 * R))**2 - 1.0     # forma cerrada
    core   = rdot2(R, M, b, Q=0.0)          # núcleo general
    assert np.allclose(closed, core, atol=1e-12), "núcleo ≠ forma cerrada"

    # Con carga, el coeficiente pasa de b a (b²−Q²)/b (verificá contra tu fuente):
    Q = 0.6
    closed_q = (a + (b**2 - Q**2) / (2 * b * R))**2 - 1.0
    core_q   = rdot2(R, M, b, Q=Q)
    assert np.allclose(closed_q, core_q, atol=1e-12), "núcleo cargado ≠ forma cerrada"

# ─────────────────────────────────────────────────────────────────────────────
# El mapa
# ─────────────────────────────────────────────────────────────────────────────
def make_map(M=1.0, a_values=(0.3, 0.5, 0.7, 0.9, 1.0, 1.2),
             R_max_plot=8.0, Q=0.0, save="initial_conditions_map.png"):
    rs = 2.0 * M
    R  = np.linspace(0.05, R_max_plot, 2000)

    # color por régimen; a=1/2 (black shell) resaltado aparte
    def color_for(a):
        if abs(a - 0.5) < 1e-9: return "#111111"          # black shell
        if a < 1.0:  return plt.cm.winter((a) / 1.0)      # ligado
        if a == 1.0: return "#E8A33D"                     # marginal
        return plt.cm.autumn((a - 1.0) / 0.6)             # no ligado

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.suptitle(
        r"Mapa de condiciones iniciales — cáscara de polvo  "
        r"$\sqrt{\dot R^2+f_-}-\sqrt{\dot R^2+f_+}=b/R$",
        fontsize=12, y=0.99)

    for a in a_values:
        b   = M / a
        col = color_for(a)
        lw  = 2.6 if abs(a - 0.5) < 1e-9 else 1.7
        rd2 = rdot2(R, M, b, Q)
        ok  = rd2 >= 0.0
        Rd  = np.sqrt(np.clip(rd2, 0, None))
        lbl = rf"$a={a:g}$  ($b={b:.2f}$) — {regime(a)}"

        # Panel 1 — retrato de fase (rama de colapso Ṙ<0 sólida, expansión punteada)
        ax1.plot(R[ok], -Rd[ok], color=col, lw=lw, label=lbl)
        ax1.plot(R[ok],  Rd[ok], color=col, lw=lw, ls=":", alpha=0.35)
        Rtp = turning_point(a, b)
        if Rtp is not None and Rtp <= R_max_plot:
            ax1.plot(Rtp, 0, "o", color=col, ms=6, zorder=5)

        # Panel 2 — potencial efectivo
        ax2.plot(R, V_eff(R, M, b, Q), color=col, lw=lw, label=lbl)

    # horizonte en ambos paneles
    for ax in (ax1, ax2):
        ax.axvline(rs, color="#C0392B", ls="--", lw=1.2, alpha=0.9)
        ax.text(rs, ax.get_ylim()[1], r"  $r=2M$", color="#C0392B",
                fontsize=9, va="top")

    # Panel 1 cosmética
    ax1.axhline(0, color="#999", lw=0.8)
    ax1.set_xlim(0, R_max_plot); ax1.set_ylim(-1.6, 1.6)
    ax1.set_xlabel(r"$R/M$"); ax1.set_ylabel(r"$\dot R$")
    ax1.set_title("Retrato de fase  (— colapso,  ⋯ expansión,  ● retorno)", fontsize=10)
    ax1.annotate("", xy=(2.6, -1.15), xytext=(4.2, -0.75),
                 arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
    ax1.text(4.3, -0.7, "colapso", color="#666", fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    # Panel 2 cosmética + región prohibida
    ax2.axhline(0, color="#999", lw=0.8)
    ax2.fill_between(R, 0, 3, color="#CCCCCC", alpha=0.35, zorder=0)
    ax2.text(R_max_plot*0.6, 0.55, r"$V>0$: prohibido ($\dot R^2<0$)",
             color="#777", fontsize=9)
    ax2.set_xlim(0, R_max_plot); ax2.set_ylim(-3, 1.2)
    ax2.set_xlabel(r"$R/M$"); ax2.set_ylabel(r"$V(R)=-\dot R^2$")
    ax2.set_title(r"Potencial efectivo  (retorno $\Leftrightarrow V=0$)", fontsize=10)

    # nota del black shell
    fig.text(0.5, 0.02,
             r"Negro grueso: $a=1/2$ (black shell, límite polvo) — el retorno cae EXACTO "
             r"en el horizonte.  $\ddot R<0$ siempre $\Rightarrow$ no es equilibrio, es "
             r"retorno momentáneo (equilibrio real requiere presión, Núñez 1997).",
             ha="center", fontsize=8.5, color="#444")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.14, wspace=0.22)
    fig.savefig(save, dpi=150)
    print(f"guardado: {save}")
    return fig


if __name__ == "__main__":
    _selftest()
    print("selftest OK — el núcleo reproduce la forma cerrada (Q=0 y Q≠0)")
    make_map()
