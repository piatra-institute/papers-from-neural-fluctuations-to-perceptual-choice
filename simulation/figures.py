"""Figures for From Neural Fluctuations to Perceptual Choice."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, AMBER, GREEN, BLUE, GRAY, RED, PURPLE = ("#1a1a1a", "#d9d9d9", "#b45309", "#15803d",
                                                    "#2563eb", "#78716c", "#b3202c", "#7c3aed")
SCEN_ORDER = ["integrator", "leak", "instability", "drift variability", "urgency", "aligned nuisance"]
SCEN_COL = [INK, BLUE, RED, AMBER, GREEN, PURPLE]


def _style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def _title(ax, t):
    ax.set_title(t, fontsize=9.5, loc="left")


def plot_relation(r, path):
    fp, dep = r["fp_curves"], r["departures"]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.7))
    ax = axes[0]
    h = np.array(fp["h"])
    cols = [INK, BLUE, AMBER, RED]
    marks = ["o", "s", "^", "D"]
    for st, c, mk in zip(fp["sets"], cols, marks):
        ax.plot(h, st["p_exact"], color=c, lw=1.2, alpha=0.8)
        ax.plot(h, st["p_mc"], mk, color=c, ms=4, mfc="none", label=f"{st['name']}  (D/(MB) = {st['theta']:.2f})")
    ax.set_xlabel("evidence h (coherence)", fontsize=9)
    ax.set_ylabel("P(choice +)", fontsize=9)
    ax.legend(fontsize=6.6, frameon=False, loc="upper left")
    _title(ax, "a. choice depends on D/(MB) only")
    _style(ax)
    ax = axes[1]
    ax.plot(dep["Lam"], dep["leak"], color=BLUE, lw=1.8)
    ax.axhline(1, color=GRAY, lw=0.8, ls=":")
    ax.axhline(2, color=GRAY, lw=0.8, ls="--")
    ax.text(5.9, 1.93, "strong-leak limit 2", fontsize=7.5, ha="right", va="top", color=GRAY)
    ax.set_xlabel("leak number  Λ = λB²/(2D)", fontsize=9)
    ax.set_ylabel("slope relative to MB/D", fontsize=9)
    _title(ax, "b. leak and instability")
    _style(ax)
    ax = axes[2]
    ax.plot(dep["s"], dep["drift"], color=AMBER, lw=1.8)
    ax.set_xlabel("drift variability  s = ηB/D", fontsize=9)
    ax.set_ylabel("slope relative to MB/D", fontsize=9)
    ax.set_ylim(0, 1.05)
    _title(ax, "c. drift variability across trials")
    _style(ax)
    ax = axes[3]
    ax.plot(dep["z"], dep["start_bias"], color=RED, lw=1.6, label="start-point offset z/B")
    ax.plot(dep["z"], dep["start_var"], color=GREEN, lw=1.6, label="start-point spread sz/B")
    ax.set_xlabel("offset or half-width relative to B", fontsize=9)
    ax.set_ylabel("slope relative to MB/D", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    _title(ax, "d. start point")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_readouts(r, path):
    f = r["fdr"]
    taus = np.array(f["taus"])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), sharey=True)
    cols = {"conjugate": BLUE, "Fisher discriminant": INK, "mean response": AMBER, "first coordinate": GREEN}
    styles = {"conjugate": "-", "Fisher discriminant": "--", "mean response": "-", "first coordinate": "-"}
    for ax, case, lab in zip(axes, ("detailed balance", "broken detailed balance"), ("a.", "b.")):
        rd = f["two_d"][case]["readouts"]
        for name, v in rd.items():
            rho = np.array(v["rho"], dtype=float)
            if not v["R_positive"]:
                continue
            ax.plot(taus, rho, styles[name], color=cols[name], lw=1.8 if name != "Fisher discriminant" else 1.4,
                    label=name)
        ax.axhline(1, color=GRAY, lw=0.6)
        ax.set_xlabel("lag τ", fontsize=9)
        ax.set_ylim(0, 3.2)
        _title(ax, f"{lab} {case}")
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        _style(ax)
    axes[0].set_ylabel("-C'(τ)/R(τ), relative to τ = 0", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_estimators(r, path):
    ex, sc = r["example"], r["scenarios"]["integrator"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    ax = axes[0]
    st, mom = np.array(ex["stime"]), np.array(ex["moment"])
    ax.plot(st, mom, "o", ms=3.2, color=INK, label="stopped split-half cross moment")
    xx = np.linspace(0, st.max(), 50)
    D = r["reference"]["D"] / r["reference"]["B"] ** 2
    ax.plot(xx, 2 * D * xx, color=GRAY, lw=1, ls="--", label=f"2D/B² = {2 * D:.1f} per s")
    ax.set_xlabel("stopped time  E[min(t, τ)]  (s)", fontsize=9)
    ax.set_ylabel("cross moment / (B₁B₂)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _title(ax, "a. diffusion from zero-evidence trials, one run")
    _style(ax)
    ax = axes[1]
    names = [("beta_pred_naive", "one-bin increments,\nsurvivors"), ("beta_pred_vario", "variogram,\nsurvivors"),
             ("beta_pred", "stopped split-half,\nsustained M"), ("beta_pred_pulse", "stopped split-half,\npulse M")]
    for i, (k, lab) in enumerate(names):
        m, s = sc[k]["mean"], sc[k]["sd"]
        ax.errorbar(i, m, yerr=s, fmt="o", color=[RED, AMBER, INK, BLUE][i], capsize=3, ms=5)
    fb = sc["beta_fit"]
    ax.axhspan(fb["mean"] - fb["sd"], fb["mean"] + fb["sd"], color=GREEN, alpha=0.15, lw=0)
    ax.axhline(fb["mean"], color=GREEN, lw=1, label="slope fitted to choices (mean ± SD)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)), [n[1] for n in names], fontsize=7.5)
    ax.set_ylabel("predicted slope MB/D (per unit coherence)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    _title(ax, "b. four estimators, pure integrator (mean ± SD over runs)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_scenarios(r, path):
    s = r["scenarios"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    ax = axes[0]
    inst = s["_theory"]["instantaneous_slope"]
    for name, c in zip(SCEN_ORDER, SCEN_COL):
        x, y = s[name]["beta_pred"], s[name]["beta_fit"]
        ax.errorbar(x["mean"], y["mean"], xerr=x["se"], yerr=y["se"], fmt="o", color=c, ms=6, capsize=2, label=name)
        ax.plot(inst, y["mean"], "o", mfc="none", color=c, ms=6)
    lo, hi = 6.5, 13.5
    ax.plot([lo, hi], [lo, hi], color=GRAY, lw=0.8)
    ax.fill_between([lo, hi], [lo / 1.1, hi / 1.1], [lo * 1.1, hi * 1.1], color=GRAY, alpha=0.12, lw=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("slope predicted from neural D, M, B", fontsize=9)
    ax.set_ylabel("slope fitted to choices", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.text(hi - 0.1, lo + 0.25, "open circles: instantaneous MB/D\nband: ±10%", fontsize=7.5, ha="right", color=GRAY)
    _title(ax, "a. prediction and behaviour, six generating models")
    _style(ax)
    ax = axes[1]
    x = np.arange(len(SCEN_ORDER))
    for j, (key, lab, off, mk) in enumerate((("late_over_early", "late / early D/(MB)", -0.12, "o"),
                                             ("M_pulse_over_sustained", "pulse / sustained M", 0.12, "s"))):
        m = [s[n][key]["mean"] for n in SCEN_ORDER]
        e = [s[n][key]["se"] for n in SCEN_ORDER]
        ax.errorbar(x + off, m, yerr=e, fmt=mk, color=[INK, BLUE][j], capsize=2, ms=5, label=lab)
    ax.axhline(1, color=GRAY, lw=0.8)
    ax.set_xticks(x, [n.replace(" ", "\n") for n in SCEN_ORDER], fontsize=7.5)
    ax.set_ylabel("ratio (mean ± SE over runs)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _title(ax, "b. two diagnostics computed without choices")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_power(r, path):
    p = r["power"]["rows"]
    tr = np.array([x["trials"] for x in p])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7))
    ax = axes[0]
    ax.plot(tr, [x["sd_log_ratio"] for x in p], "o-", color=INK, label="log(fitted × D/(MB))")
    ax.plot(tr, [x["sd_log_beta_pred"] for x in p], "s--", color=BLUE, label="log predicted slope")
    ax.plot(tr, [x["sd_log_beta_fit"] for x in p], "^--", color=GREEN, label="log fitted slope")
    ref = r["power"]["kappa"] / np.sqrt(tr)
    ax.plot(tr, ref, color=GRAY, lw=0.8, ls=":", label="pooled κ/√trials")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("trials (zero, pulse, low-evidence and graded sets)", fontsize=9)
    ax.set_ylabel("standard deviation across runs", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    _title(ax, "a. precision of the comparison")
    _style(ax)
    ax = axes[1]
    tt = np.geomspace(tr.min(), tr.max() * 2, 60)
    from scipy import stats as _st
    for m, c in ((10, RED), (25, INK)):
        sdp = r["power"]["kappa"] / np.sqrt(tt)
        ax.plot(tt, np.clip(2 * _st.norm.cdf(np.log(1 + m / 100) / sdp - _st.norm.ppf(0.95)) - 1, 0, 1), color=c, lw=1.6,
                label=f"margin ±{m}%, pooled SD")
        ax.plot(tr, [x[f"tost_power_{m}"] for x in p], "o", color=c, mfc="none", ms=5, label=f"margin ±{m}%, SD at each size")
    ax.axhline(0.8, color=GRAY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("trials", fontsize=9)
    ax.set_ylabel("power of the equivalence test (true ratio 1)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    _title(ax, "b. two one-sided tests at level 0.05")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_all(r, out):
    out = Path(out)
    plot_relation(r, out / "relation.png")
    plot_readouts(r, out / "readouts.png")
    plot_estimators(r, out / "estimators.png")
    plot_scenarios(r, out / "scenarios.png")
    plot_power(r, out / "power.png")
