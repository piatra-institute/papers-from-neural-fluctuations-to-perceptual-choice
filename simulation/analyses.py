"""Analyses for "From Neural Fluctuations to Perceptual Choice".

A bounded accumulator dX = (M h(t) - lam X) dt + sqrt(2 D) dW, X(0) = z, stops at +-B. Sections:
  theory     first-passage choice probabilities and the factors by which departures from the pure integrator
             change the psychometric slope (exact formulas checked by Monte Carlo)
  fdr        fluctuation-response ratios of stationary linear systems along different readouts
  neural     a spiking population that carries the accumulator, the estimators of D, M and B, and the
             comparison of the predicted slope M B / D with the slope fitted to choices
  power      sample sizes for the equivalence test between neural prediction and behaviour
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
from scipy import integrate, linalg, optimize, special, stats

# reference parameters: decision variable in units of the bound, evidence in units of motion coherence
REF = dict(D=0.8, B=1.0, M=8.0)          # 1/s, bound, per coherence per s; slope M B / D = 10 per unit coherence
COHS = np.array([0.0, 0.032, 0.064, 0.128, 0.256, 0.512])


def logistic(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


# ========================================================================================================= theory

def p_plus(v, D, B, z=0.0):
    """Probability that dX = v dt + sqrt(2D) dW started at z exits (-B, B) through +B."""
    a = v / D
    if abs(a) < 1e-12:
        return (z + B) / (2 * B)
    return float(np.expm1(-a * (z + B)) / np.expm1(-2 * a * B))


def p_plus_leak(v, lam, D, B, z=0.0):
    """Exit probability through +B for dX = (v - lam X) dt + sqrt(2D) dW (scale-function quadrature)."""
    phi = lambda y: -(v * y - 0.5 * lam * y * y) / D
    ys = np.linspace(-B, B, 4001)
    shift = np.max(phi(ys))
    s = lambda y: np.exp(phi(y) - shift)
    num = integrate.quad(s, -B, z, limit=200, epsabs=0, epsrel=1e-11)[0]
    den = integrate.quad(s, -B, B, limit=200, epsabs=0, epsrel=1e-11)[0]
    return num / den


def leak_factor(Lam):
    """Slope at h = 0 with leak relative to the pure integrator, Lam = lam B^2 / (2 D): (e^Lam - 1)/(Lam int_0^1 e^{Lam u^2})."""
    if abs(Lam) < 1e-10:
        return 1.0
    I = integrate.quad(lambda u: np.exp(Lam * u * u), 0, 1, epsabs=0, epsrel=1e-12)[0]
    return float(np.expm1(Lam) / (Lam * I))


def drift_var_factor(s, n=80):
    """Slope factor 4 E[logistic'(s Z)], Z standard normal, s = eta B / D (eta = SD of drift across trials)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    f = logistic(s * x)
    return float(4 * np.sum(w * f * (1 - f)) / np.sqrt(2 * np.pi))


def start_bias_factor(zb):
    """Slope factor at h = 0 for a start point z = zb * B: 1 - zb^2."""
    return 1.0 - zb ** 2


def start_var_factor(szb):
    """Slope factor for a start point uniform on (-sz, sz), szb = sz / B: 1 - szb^2 / 3."""
    return 1.0 - szb ** 2 / 3.0


def local_slope(P, h=1e-4):
    """beta_0 = 4 dP/dh at h = 0 (equals the logistic slope when P is logistic)."""
    return 4 * (P(h) - P(-h)) / (2 * h)


def simulate_accumulator(h, n, rng, D=0.8, B=1.0, M=8.0, lam=0.0, eta=0.0, z=0.0, sz=0.0, dt=1e-3, T=6.0,
                         bound=None, gain=None):
    """Monte Carlo of the bounded accumulator with a Brownian-bridge test for crossings between steps.

    h may be a scalar evidence level. eta: SD of the drift across trials; sz: half-width of a uniform start point.
    bound(t): optional time-varying bound. gain(t): optional multiplicative gain on the represented variable
    Z = g(t) X with a fixed bound on Z (an urgency signal). Returns choice (+1/-1) and decision time."""
    x = np.full(n, float(z)) + (rng.uniform(-sz, sz, n) if sz > 0 else 0.0)
    v = M * h + (eta * rng.standard_normal(n) if eta > 0 else 0.0)
    choice = np.zeros(n)
    tdec = np.full(n, np.nan)
    alive = np.ones(n, dtype=bool)
    sq = np.sqrt(2 * D * dt)
    steps = int(round(T / dt))
    for k in range(steps):
        t = k * dt
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        xi = x[idx]
        vi = v[idx] if np.ndim(v) else v
        if gain is None:
            b0 = B if bound is None else bound(t)
            b1 = B if bound is None else bound(t + dt)
            xn = xi + (vi - lam * xi) * dt + sq * rng.standard_normal(idx.size)
            var = 2 * D * dt
        else:
            g0, g1 = gain(t), gain(t + dt)
            gd = (g1 - g0) / dt
            xn = xi + (gd / g0 * xi + g0 * vi) * dt + g0 * sq * rng.standard_normal(idx.size)
            b0 = b1 = B
            var = 2 * D * dt * g0 * g0
        up = xn >= b1
        dn = xn <= -b1
        mid = ~(up | dn)
        pu = np.exp(-2 * np.clip(b0 - xi, 0, None) * np.clip(b1 - xn, 0, None) / var)
        pd = np.exp(-2 * np.clip(xi + b0, 0, None) * np.clip(xn + b1, 0, None) / var)
        u = rng.random(idx.size)
        up |= mid & (u < pu)
        dn |= mid & ~up & (u < pu + pd) & (u >= pu)
        hit = up | dn
        choice[idx[up]] = 1
        choice[idx[dn & ~up]] = -1
        tdec[idx[hit]] = t + dt
        x[idx] = xn
        alive[idx[hit]] = False
    # undecided at T: choose by sign (rare at the reference parameters)
    rest = np.flatnonzero(alive)
    choice[rest] = np.where(x[rest] >= 0, 1, -1)
    tdec[rest] = T
    return choice, tdec


def fit_logistic(h, y):
    """Maximum-likelihood logistic regression of y in {0,1} on h with intercept; returns slope, SE, bias."""
    X = np.column_stack([np.ones_like(h), h])
    b = np.zeros(2)
    for _ in range(50):
        p = logistic(X @ b)
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    cov = np.linalg.inv(H)
    return float(b[1]), float(np.sqrt(cov[1, 1])), float(b[0])


def theory_checks(seed=1, n=40000):
    rng = np.random.default_rng(seed)
    D, B, M = REF["D"], REF["B"], REF["M"]
    out = {"params": dict(D=D, B=B, M=M, slope=M * B / D, theta=D / (M * B))}
    # 1. pure integrator: exit probabilities against the closed form
    rows, zs = [], []
    for h in (0.0, 0.032, 0.064, 0.128, -0.064):
        c, _ = simulate_accumulator(h, n, rng)
        p_mc = float(np.mean(c > 0))
        p_th = p_plus(M * h, D, B)
        z = (p_mc - p_th) / np.sqrt(p_th * (1 - p_th) / n)
        rows.append({"h": h, "p_mc": p_mc, "p_exact": p_th, "zscore": float(z)})
        zs.append(abs(z))
    out["ddm"] = {"rows": rows, "max_abs_zscore": float(max(zs)), "n": n}
    # 2. invariance: rescaling the coordinate by c and time by k leaves D/(M B) unchanged
    c_, k_ = 7.3, 0.25
    D2, M2, B2 = c_ ** 2 * D / k_, c_ * M / k_, c_ * B
    out["invariance"] = {"c": c_, "k": k_, "theta": D / (M * B), "theta_rescaled": D2 / (M2 * B2),
                         "p_rescaled": p_plus(M2 * 0.064, D2, B2), "p": p_plus(M * 0.064, D, B)}
    # 3. leak factor: closed form against quadrature and Monte Carlo
    lrows = []
    for Lam in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0):
        lam = Lam * 2 * D / B ** 2
        quad_slope = local_slope(lambda hh: p_plus_leak(M * hh, lam, D, B)) / (M * B / D)
        lrows.append({"Lam": Lam, "lam": lam, "factor": leak_factor(Lam), "factor_quad": quad_slope})
    out["leak"] = {"rows": lrows, "limit_strong_leak": 2.0,
                   "max_err": float(max(abs(r["factor"] - r["factor_quad"]) for r in lrows))}
    # Monte Carlo at Lam = 1 and Lam = -1: exit probability at h = 0.064
    mc = []
    for Lam in (1.0, -1.0):
        lam = Lam * 2 * D / B ** 2
        c, _ = simulate_accumulator(0.064, n, rng, lam=lam)
        p_mc = float(np.mean(c > 0))
        p_th = p_plus_leak(M * 0.064, lam, D, B)
        mc.append({"Lam": Lam, "p_mc": p_mc, "p_exact": p_th, "zscore": float((p_mc - p_th) / np.sqrt(p_th * (1 - p_th) / n))})
    out["leak"]["mc"] = mc
    # 4. drift variability and start-point variability against Monte Carlo
    dv = []
    for s in (0.5, 1.0, 2.0):
        dv.append({"s": s, "factor": drift_var_factor(s)})
    eta = 1.0 * D / B                         # s = 1
    c, _ = simulate_accumulator(0.064, n, rng, eta=eta)
    x, w = np.polynomial.hermite_e.hermegauss(80)
    p_th = float(np.sum(w * logistic((M * 0.064 + eta * x) * B / D)) / np.sqrt(2 * np.pi))
    p_mc = float(np.mean(c > 0))
    out["drift_var"] = {"rows": dv, "mc": {"s": 1.0, "p_mc": p_mc, "p_exact": p_th,
                                           "zscore": float((p_mc - p_th) / np.sqrt(p_th * (1 - p_th) / n))}}
    sv = [{"sz": s, "factor": start_var_factor(s)} for s in (0.2, 0.4, 0.6)]
    c, _ = simulate_accumulator(0.064, n, rng, sz=0.6)
    p_th = integrate.quad(lambda zz: p_plus(M * 0.064, D, B, zz), -0.6, 0.6)[0] / 1.2
    p_mc = float(np.mean(c > 0))
    out["start_var"] = {"rows": sv, "mc": {"sz": 0.6, "p_mc": p_mc, "p_exact": p_th,
                                           "zscore": float((p_mc - p_th) / np.sqrt(p_th * (1 - p_th) / n))}}
    # 5. start bias: slope at h = 0 and the closed form for the whole curve
    sb = [{"zb": zb, "factor": start_bias_factor(zb),
           "factor_numeric": local_slope(lambda hh: p_plus(M * hh, D, B, zb * B)) / (M * B / D)} for zb in (0.1, 0.2, 0.3)]
    c, _ = simulate_accumulator(0.064, n, rng, z=0.3)
    p_th = p_plus(M * 0.064, D, B, 0.3)
    p_mc = float(np.mean(c > 0))
    out["start_bias"] = {"rows": sb, "mc": {"zb": 0.3, "p_mc": p_mc, "p_exact": p_th,
                                            "zscore": float((p_mc - p_th) / np.sqrt(p_th * (1 - p_th) / n))}}
    return out


def collapsing_bounds(seed=2, n=66000):
    """Psychometric slope (logistic fit over the coherence set) against the naive M B0 / D and M Bbar / D, where Bbar is
    the mean bound at the time of decision, for hyperbolically collapsing bounds B(t) = B0 (1 - t / (t + t50))."""
    rng = np.random.default_rng(seed)
    D, B, M = REF["D"], REF["B"], REF["M"]
    rows = []
    hs = np.concatenate([-COHS[:0:-1], COHS])
    for t50 in (np.inf, 2.0, 1.0, 0.5):
        bound = None if np.isinf(t50) else (lambda t, t50=t50: B * (1 - t / (t + t50)))
        H, Y, BB = [], [], []
        for h in hs:
            c, td = simulate_accumulator(h, n // len(hs), rng, bound=bound, T=8.0)
            H.append(np.full(c.size, h))
            Y.append((c > 0).astype(float))
            BB.append(B * np.ones(c.size) if bound is None else bound(td))
        H, Y, BB = map(np.concatenate, (H, Y, BB))
        slope, se, _ = fit_logistic(H, Y)
        bbar = float(np.mean(BB[np.abs(H) < 0.1]))
        rows.append({"t50": None if np.isinf(t50) else t50, "slope_fit": slope, "se": se,
                     "naive_B0": M * B / D, "naive_Bbar": M * bbar / D, "Bbar": bbar})
    return rows


# ============================================================================================================ fdr

def fr_ratio(K, S, f, w, taus):
    """Fluctuation-response ratio rho(tau) = -dC_w/dtau / R_w for dx = (-K x + f h) dt + noise with stationary covariance S.
    C_w(tau) = w' e^{-K tau} S w, R_w(tau) = w' e^{-K tau} f."""
    rho, R = [], []
    for t in taus:
        E = linalg.expm(-K * t)
        r = float(w @ E @ f)
        rho.append(float(w @ K @ E @ S @ w) / r)
        R.append(r)
    return np.array(rho), np.array(R)


def fdr_linear():
    """Stationary linear systems: the ratio is constant in tau along the conjugate readout under detailed balance and
    varies otherwise; under detailed balance the Fisher discriminant of the stationary response is the conjugate readout."""
    taus = np.linspace(0, 3, 121)
    out = {"taus": taus}
    # one dimension
    lam, D1, M1 = 2.0, 0.8, 8.0
    rho, _ = fr_ratio(np.array([[lam]]), np.array([[D1 / lam]]), np.array([M1]), np.array([1.0]), taus)
    out["ou"] = {"lam": lam, "D": D1, "M": M1, "D_over_M": D1 / M1, "max_rel_dev": float(np.max(np.abs(rho / (D1 / M1) - 1)))}
    # two dimensions with the same stationary covariance, with and without detailed balance
    S = np.array([[1.0, 0.6], [0.6, 1.5]])
    Dm = np.array([[1.0, 0.3], [0.3, 0.5]])
    Q = 0.6 * np.array([[0.0, 1.0], [-1.0, 0.0]])
    f = np.array([1.0, 0.2])
    Si = np.linalg.inv(S)
    cases = {}
    for name, K in (("detailed balance", Dm @ Si), ("broken detailed balance", (Dm + Q) @ Si)):
        lyap = K @ S + S @ K.T - 2 * Dm
        conj = np.linalg.solve(Dm, f)
        lda = Si @ np.linalg.solve(K, f)
        mean_dir = np.linalg.solve(K, f)
        e1 = np.array([1.0, 0.0])
        cos = float(conj @ lda / np.linalg.norm(conj) / np.linalg.norm(lda))
        readouts = {}
        # identity: S w* = K^{-1} f for the Fisher discriminant w* = S^{-1} K^{-1} f, so -C' = R along w*
        out.setdefault("identity_residual", 0.0)
        out["identity_residual"] = max(out["identity_residual"], float(np.max(np.abs(S @ lda - np.linalg.solve(K, f)))))
        for rname, w in (("conjugate", conj), ("Fisher discriminant", lda), ("mean response", mean_dir), ("first coordinate", e1)):
            rho, R = fr_ratio(K, S, f, w, taus)
            ok = np.all(R > 0.02 * R[0])
            readouts[rname] = {"rho": rho / rho[0], "min_R_ratio": float(np.min(R / R[0])),
                               "max_rel_dev": float(np.max(np.abs(rho / rho[0] - 1))) if ok else None,
                               "R_positive": bool(ok)}
        cases[name] = {"lyapunov_residual": float(np.max(np.abs(lyap))), "cos_conj_lda": cos,
                       "angle_deg": float(np.degrees(np.arccos(min(1.0, abs(cos))))), "readouts": readouts,
                       "eig_real": [float(x) for x in np.real(np.linalg.eigvals(K))]}
    out["two_d"] = cases
    out["S"], out["Dm"], out["Q"], out["f"] = S, Dm, Q, f
    return out


# ========================================================================================================= neural

SCENARIOS = {
    "integrator": {},
    "leak": {"lam": 1.6},                       # Lam = lam B^2 / (2 D) = 1
    "instability": {"lam": -1.6},               # Lam = -1
    "drift variability": {"eta": 0.8},          # s = eta B / D = 1
    "urgency": {"gain_tau": 1.0},               # represented variable Z = (1 + t / gain_tau) X, fixed bound on Z
    "aligned nuisance": {"nuis_sd": 0.3, "nuis_tau": 1.0, "nuis_align": 1.0},   # slow signal along the coding axis
}

DESIGN = dict(n_train=400, n_zero=1000, n_pulse=2000, n_neural=2000, n_graded=3000, pulse_dh=0.2, pulse_dur=0.1,
              pulse_on=(0.1, 0.5), tnd=0.3, T=2.5, dt=2e-3, bin=0.02, N=200)


def simulate_latent(h0, sgn, on, rng, sc, d):
    """Represented decision variable Z averaged in bins, frozen at +-B after the decision; nuisance Y in bins.
    h0: evidence per trial; sgn, on: pulse sign (0 for none) and onset per trial."""
    D, B, M = REF["D"], REF["B"], REF["M"]
    lam, eta = sc.get("lam", 0.0), sc.get("eta", 0.0)
    gtau = sc.get("gain_tau")
    n = h0.size
    dt, T, bw = d["dt"], d["T"], d["bin"]
    steps, per = int(round(T / dt)), int(round(bw / dt))
    nb = steps // per
    x = np.zeros(n)
    v0 = M * h0 + (eta * rng.standard_normal(n) if eta > 0 else 0.0)
    zb = np.zeros((n, nb))
    acc = np.zeros(n)
    choice = np.zeros(n)
    tau = np.full(n, np.nan)
    alive = np.ones(n, dtype=bool)
    frozen = np.zeros(n)
    sq = np.sqrt(2 * D * dt)
    ys = None
    if sc.get("nuis_sd"):
        a = np.exp(-dt / sc["nuis_tau"])
        y = sc["nuis_sd"] * rng.standard_normal(n)
        yb = np.zeros((n, nb))
        yacc = np.zeros(n)
    for k in range(steps):
        t = k * dt
        hp = np.where((sgn != 0) & (t >= on) & (t < on + d["pulse_dur"]), sgn * d["pulse_dh"], 0.0)
        v = v0 + M * hp
        if gtau is None:
            g0 = g1 = 1.0
            drift = v - lam * x
        else:
            g0, g1 = 1 + t / gtau, 1 + (t + dt) / gtau
            drift = (1 / gtau) / g0 * x + g0 * v
        xn = x + drift * dt + g0 * sq * rng.standard_normal(n)
        var = 2 * D * dt * g0 * g0
        up = alive & (xn >= B)
        dn = alive & (xn <= -B)
        mid = alive & ~(up | dn)
        pu = np.exp(-2 * np.clip(B - x, 0, None) * np.clip(B - xn, 0, None) / var)
        pd = np.exp(-2 * np.clip(x + B, 0, None) * np.clip(xn + B, 0, None) / var)
        u = rng.random(n)
        up |= mid & (u < pu)
        dn |= mid & ~up & (u >= pu) & (u < pu + pd)
        hit = up | dn
        frozen[up] = B
        frozen[dn & ~up] = -B
        choice[up] = 1
        choice[dn & ~up] = -1
        tau[hit] = t + dt
        alive &= ~hit
        x = np.where(alive, xn, frozen)
        acc += x
        if sc.get("nuis_sd"):
            y = a * y + sc["nuis_sd"] * np.sqrt(1 - a * a) * rng.standard_normal(n)
            yacc += y
        if (k + 1) % per == 0:
            j = (k + 1) // per - 1
            zb[:, j] = acc / per
            acc[:] = 0
            if sc.get("nuis_sd"):
                yb[:, j] = yacc / per
                yacc[:] = 0
    rest = alive
    choice[rest] = np.where(x[rest] >= 0, 1, -1)
    tau[rest] = np.inf
    return zb, (yb if sc.get("nuis_sd") else None), choice, tau


def make_population(rng, N):
    """Baseline rates 15-35 Hz; gains on the decision variable and on a nuisance variable proportional to the
    baseline and bounded so that rates stay positive over the range of the decision variable."""
    r0 = rng.uniform(15, 35, N)
    G = 0.5 * r0 * np.clip(rng.standard_normal(N), -1.6, 1.6)
    U = 0.5 * r0 * np.clip(rng.standard_normal(N), -1.6, 1.6)
    return r0, G, U


def rates(pop, zb, yb):
    r0, G, U = pop
    r = r0[None, None, :] + zb[:, :, None] * G[None, None, :]
    if yb is not None:
        r = r + yb[:, :, None] * U[None, None, :]
    return np.clip(r, 0.5, None)


def train_decoders(counts, h0, bw, halves, alpha=10.0):
    """Shrinkage linear discriminant on bin counts averaged over 0.1-0.3 s, strong-evidence trials only."""
    j0, j1 = int(0.1 / bw), int(0.3 / bw)
    F = counts[:, j0:j1, :].mean(axis=1)
    out = []
    for idx in [np.arange(F.shape[1])] + list(halves):
        X = F[:, idx]
        mp, mn = X[h0 > 0].mean(0), X[h0 < 0].mean(0)
        C = 0.5 * (np.cov(X[h0 > 0].T) + np.cov(X[h0 < 0].T))
        C = C + alpha * np.mean(np.diag(C)) * np.eye(C.shape[0])
        w = np.zeros(F.shape[1])
        w[idx] = np.linalg.solve(C, mp - mn)
        out.append(w / bw)
    return out          # full, half 1, half 2; projection A = w . counts gives rate-weighted units


def project(pop, zb, yb, ws, rng, bw, exact=True):
    """Decoded variables A_k = w_k . counts for each decoder; exact Poisson counts or a Gaussian approximation."""
    if exact:
        counts = rng.poisson(rates(pop, zb, yb) * bw)
        return [counts @ w for w in ws], None
    r0, G, U = pop
    outs = []
    for w in ws:
        mu = bw * (w @ r0 + (w @ G) * zb + ((w @ U) * yb if yb is not None else 0.0))
        var = bw * ((w * w) @ r0 + ((w * w) @ G) * zb + (((w * w) @ U) * yb if yb is not None else 0.0))
        outs.append(mu + np.sqrt(np.clip(var, 1e-9, None)) * rng.standard_normal(mu.shape))
    return outs, None


def _bin_avg(fn, nb, bw, k=5):
    """Average of fn(t) over each bin (k sub-points); fn returns (trials, len(t))."""
    sub = (np.arange(nb)[:, None] + (np.arange(k)[None, :] + 0.5) / k) * bw
    vals = fn(sub.ravel())
    return vals.reshape(vals.shape[0], nb, k).mean(axis=2)


def stopped(A, tau_hat, saccade, bw):
    """A at min(t, tau_hat) in bins: bins that end before tau_hat keep their value, later bins (including the bin in
    which tau_hat falls) take a bin from the post-decision period, 0.1 s before the saccade."""
    nb = A.shape[1]
    tend = (np.arange(nb) + 1) * bw
    fin = np.isfinite(saccade)
    js = np.full(saccade.shape, nb - 1)
    js[fin] = np.clip(np.floor((saccade[fin] - 0.1) / bw).astype(int), 0, nb - 1)
    frozen_val = A[np.arange(A.shape[0]), js]
    return np.where(tend[None, :] <= tau_hat[:, None], A, frozen_val[:, None])


def stop_time(tau_cap, nb, bw):
    """Time at which the stopped value of each bin is taken: the bin centre for bins that end before the decision,
    the decision time for the bin in which it falls and all later bins. Also returns the pre-decision mask."""
    tc = (np.arange(nb) + 0.5) * bw
    tend = tc + 0.5 * bw
    pre = tend[None, :] <= tau_cap[:, None]
    return np.where(pre, tc[None, :], tau_cap[:, None]), pre


def _slope(xv, yv, mask, intercept=True):
    if intercept:
        X = np.column_stack([np.ones(mask.sum()), xv[mask]])
        return float(np.linalg.lstsq(X, yv[mask], rcond=None)[0][1])
    return float(np.sum(xv[mask] * yv[mask]) / np.sum(xv[mask] ** 2))


def estimate(data, d, tnd_hat=None):
    """Neural estimates of D, M and B, and the slope M B / D they predict, against the slope fitted to choices.

    D: zero-evidence trials, split-half cross moment of the stopped decoded variable against E[min(t, tau)].
    M: low-evidence trials, stopped mean displacement against E[h^2 min(t, tau)] (sustained), and pulse trials,
       onset-aligned stopped displacement against the stopped pulse exposure (pulse).
    B: half the difference between the decoded levels before the saccade for the two choices."""
    bw = d["bin"]
    tnd_hat = d["tnd"] if tnd_hat is None else tnd_hat
    nb = data["zero"]["A"].shape[1]
    tc = (np.arange(nb) + 0.5) * bw
    tend = (np.arange(nb) + 1) * bw

    def taus(q):
        sac = q["tau"] + d["tnd"]
        th = sac - tnd_hat
        return sac, th, np.where(np.isfinite(th), th, d["T"])

    # ---- D from zero-evidence trials
    z = data["zero"]
    sac, th, cap = taus(z)
    S = {k: stopped(z[k], th, sac, bw) for k in ("A", "A1", "A2")}
    mom = np.mean((S["A1"] - S["A1"][:, :1]) * (S["A2"] - S["A2"][:, :1]), axis=0)
    st, pre_z = stop_time(cap, nb, bw)
    stime = (st - pre_z * bw / 6.0).mean(0)          # E[bin average^2] of a Brownian path is 2D (t_c - bw/6)
    Dw = {w: _slope(stime, mom, (tc >= w[0]) & (tc <= w[1])) / 2 for w in ((0.1, 0.6), (0.1, 0.3), (0.3, 0.6))}
    mom_full = np.mean((S["A"] - S["A"][:, :1]) ** 2, axis=0)
    D_full_single = _slope(stime, mom_full, (tc >= 0.1) & (tc <= 0.6)) / 2
    A = z["A"]
    pre = (tend[None, :] <= th[:, None]) & (tc[None, :] >= 0.05) & (tc[None, :] <= 0.6)
    pre2 = pre[:, 1:] & pre[:, :-1]
    D_naive = float(np.var(np.diff(A, axis=1)[pre2]) / (2 * bw))
    V = [np.mean((A[:, k:] - A[:, :-k])[pre[:, k:] & pre[:, :-k]] ** 2) for k in range(1, 5)]
    D_vario = float(np.polyfit(np.arange(1, 5) * bw, V, 1)[0] / 2)
    # ---- M from sustained low evidence
    g = data["neural"]
    sac, th, cap = taus(g)
    Ms = {}
    st_g, _ = stop_time(cap, nb, bw)
    xs = (g["h0"][:, None] ** 2 * st_g).mean(0)
    for k in ("A", "A1", "A2"):
        Sg = stopped(g[k], th, sac, bw)
        ys = np.mean(g["h0"][:, None] * (Sg - Sg[:, :1]), axis=0)
        Ms[k] = {w: _slope(xs, ys, (tc >= w[0]) & (tc <= w[1])) for w in ((0.1, 0.6), (0.1, 0.3), (0.3, 0.6))}
    # ---- M from pulses, aligned to the onset
    p = data["pulse"]
    sac, th, cap = taus(p)
    jon = np.round(p["on"] / bw).astype(int)
    L = np.arange(0, int(round(0.3 / bw)) + 1)
    rows = np.arange(len(jon))
    Mp = {}
    st_p, _ = stop_time(cap, nb, bw)
    tl = st_p[rows[:, None], np.clip(jon[:, None] + L[None, :], 0, nb - 1)]
    ex = np.clip(tl - p["on"][:, None], 0, d["pulse_dur"]) * d["pulse_dh"]
    for k in ("A", "A1", "A2"):
        Sp = stopped(p[k], th, sac, bw)
        base = Sp[rows, np.maximum(jon - 1, 0)]
        y = p["sgn"][:, None] * (Sp[rows[:, None], np.clip(jon[:, None] + L[None, :], 0, nb - 1)] - base[:, None])
        Mp[k] = _slope(ex.mean(0), y.mean(0), np.ones(L.size, bool), intercept=False)
    # naive M: survivors' displacement 0.1 s after the pulse ends
    Ap = p["A"]
    jend = jon + int(round((d["pulse_dur"] + 0.1) / bw))
    alive = th > (jend + 1) * bw
    disp = Ap[rows, np.clip(jend, 0, nb - 1)] - Ap[rows, np.maximum(jon - 1, 0)]
    M_naive = float(np.mean((p["sgn"] * disp)[alive]) / (d["pulse_dh"] * d["pulse_dur"]))
    # ---- B from the decoded level 0.1 s before the saccade
    Bs = {}
    for k in ("A", "A1", "A2"):
        lv, ch = [], []
        for s in ("zero", "pulse", "neural", "graded"):
            q = data[s]
            sac = q["tau"] + d["tnd"]
            ok = np.isfinite(sac)
            js = np.clip(np.floor((sac[ok] - 0.1) / bw).astype(int), 0, nb - 1)
            lv.append(q[k][ok][np.arange(ok.sum()), js])
            ch.append(q["choice"][ok])
        lv, ch = np.concatenate(lv), np.concatenate(ch)
        Bs[k] = float(0.5 * (lv[ch > 0].mean() - lv[ch < 0].mean()))
    # ---- behaviour
    gb = data["graded"]
    beta, beta_se, _ = fit_logistic(gb["h0"], (gb["choice"] > 0).astype(float))

    def theta(Dv, M1, M2):
        return 0.5 * (Dv / (M1 * Bs["A2"]) + Dv / (M2 * Bs["A1"]))
    w0 = (0.1, 0.6)
    th_s = theta(Dw[w0], Ms["A1"][w0], Ms["A2"][w0])
    th_p = theta(Dw[w0], Mp["A1"], Mp["A2"])
    th_e = theta(Dw[(0.1, 0.3)], Ms["A1"][(0.1, 0.3)], Ms["A2"][(0.1, 0.3)])
    th_l = theta(Dw[(0.3, 0.6)], Ms["A1"][(0.3, 0.6)], Ms["A2"][(0.3, 0.6)])
    return {"beta_fit": beta, "beta_se": beta_se,
            "beta_pred": 1 / th_s, "ratio": beta * th_s, "log_ratio": float(np.log(beta * th_s)),
            "beta_pred_pulse": 1 / th_p, "ratio_pulse": beta * th_p,
            "beta_pred_naive": Bs["A"] * M_naive / D_naive, "beta_pred_vario": Bs["A"] * Ms["A"][w0] / D_vario,
            "beta_pred_single": Bs["A"] * Ms["A"][w0] / D_full_single,
            "theta_early": th_e, "theta_late": th_l, "late_over_early": th_l / th_e,
            "M_pulse_over_sustained": 0.5 * (Mp["A1"] / Ms["A1"][w0] + Mp["A2"] / Ms["A2"][w0]),
            "D_over_B2": Dw[w0] / (Bs["A1"] * Bs["A2"]), "M_over_B": 0.5 * (Ms["A1"][w0] / Bs["A1"] + Ms["A2"][w0] / Bs["A2"]),
            "Mp_over_B": 0.5 * (Mp["A1"] / Bs["A1"] + Mp["A2"] / Bs["A2"]), "B1": Bs["A1"], "B2": Bs["A2"]}


def run_pipeline(scen, seed, d=DESIGN, exact=True, scale=1.0, N=None, pop_seed=None):
    rng = np.random.default_rng(seed)
    prng = rng if pop_seed is None else np.random.default_rng(pop_seed)
    sc = SCENARIOS[scen]
    if N is not None:
        d = dict(d, N=N)
    pop = make_population(prng, d["N"])
    if sc.get("nuis_align") is not None:
        a = sc["nuis_align"]
        pop = (pop[0], pop[1], a * pop[1] + np.sqrt(1 - a * a) * pop[2])
    perm = prng.permutation(d["N"])
    halves = (np.sort(perm[: d["N"] // 2]), np.sort(perm[d["N"] // 2:]))
    bw = d["bin"]
    # decoder training on strong evidence (always exact Poisson counts)
    nt = d["n_train"]
    h0 = np.where(np.arange(nt) % 2 == 0, 0.512, -0.512)
    zb, yb, ch, tau = simulate_latent(h0, np.zeros(nt), np.zeros(nt), prng, sc, d)
    counts = prng.poisson(rates(pop, zb, yb) * bw)
    ws = train_decoders(counts, h0, bw, halves)
    data = {}
    sizes = {"zero": int(d["n_zero"] * scale), "pulse": int(d["n_pulse"] * scale), "neural": int(d["n_neural"] * scale),
             "graded": int(d["n_graded"] * scale)}
    levels = np.concatenate([-COHS[:0:-1], COHS])
    low = np.array([-0.256, -0.128, -0.064, 0.064, 0.128, 0.256])
    for s, n in sizes.items():
        sgn = np.zeros(n)
        on = np.zeros(n)
        if s == "graded":
            h0 = levels[np.arange(n) % levels.size]
        elif s == "neural":
            h0 = low[np.arange(n) % low.size]
        else:
            h0 = np.zeros(n)
        if s == "pulse":
            sgn = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
            on = bw * rng.integers(int(d["pulse_on"][0] / bw), int(d["pulse_on"][1] / bw) + 1, n)
        A_all = {"A": [], "A1": [], "A2": []}
        zb_l, ch_l, tau_l = [], [], []
        for c0 in range(0, n, 1000):
            sl = slice(c0, min(n, c0 + 1000))
            zb, yb, ch, tau = simulate_latent(h0[sl], sgn[sl], on[sl], rng, sc, d)
            As, _ = project(pop, zb, yb, ws, rng, bw, exact=exact)
            for key, Ak in zip(("A", "A1", "A2"), As):
                A_all[key].append(Ak)
            ch_l.append(ch)
            tau_l.append(tau)
        data[s] = {k: np.concatenate(v) for k, v in A_all.items()}
        data[s].update({"choice": np.concatenate(ch_l), "tau": np.concatenate(tau_l), "h0": h0, "sgn": sgn, "on": on})
    return data



# ========================================================================================================= summaries

KEYS = ("beta_fit", "beta_pred", "log_ratio", "beta_pred_pulse", "ratio_pulse", "beta_pred_naive", "beta_pred_vario",
        "late_over_early", "M_pulse_over_sustained", "D_over_B2", "M_over_B", "Mp_over_B")


def _summ(vals):
    out = {}
    for k in vals[0]:
        v = np.array([x[k] for x in vals], dtype=float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "se": float(v.std(ddof=1) / np.sqrt(v.size))}
    return out


def scenario_table(nrep=48, seed0=1000):
    D, B, M = REF["D"], REF["B"], REF["M"]
    out = {}
    for sc in SCENARIOS:
        vals = []
        for i in range(nrep):
            r = estimate(run_pipeline(sc, seed0 + i, exact=False), DESIGN)
            vals.append({k: r[k] for k in KEYS})
        s = _summ(vals)
        s["frac_within_25pct"] = float(np.mean([abs(x["log_ratio"]) < np.log(1.25) for x in vals]))
        s["ratio_of_means"] = s["beta_fit"]["mean"] / s["beta_pred"]["mean"]
        out[sc] = s
    # exact theory where it exists: slope at h = 0 relative to M B / D
    out["_theory"] = {"integrator": 1.0, "leak": leak_factor(1.6 * B ** 2 / (2 * D)),
                      "instability": leak_factor(-1.6 * B ** 2 / (2 * D)), "drift variability": drift_var_factor(0.8 * B / D),
                      "instantaneous_slope": M * B / D}
    out["_nrep"] = nrep
    return out


def power_analysis(scales=(0.5, 1.0, 2.0, 4.0), nrep=96, seed0=2000):
    rows = []
    for sc_ in scales:
        vals = []
        for i in range(nrep):
            r = estimate(run_pipeline("integrator", seed0 + i + int(1000 * sc_), exact=False, scale=sc_), DESIGN)
            vals.append(r)
        lr = np.array([x["log_ratio"] for x in vals])
        comp = {k: float(np.std(np.log([x[k] for x in vals]), ddof=1)) for k in ("beta_fit", "beta_pred")}
        sd = float(lr.std(ddof=1))
        trials = int(sc_ * (DESIGN["n_zero"] + DESIGN["n_pulse"] + DESIGN["n_neural"] + DESIGN["n_graded"]))
        row = {"scale": sc_, "trials": trials, "mean_log_ratio": float(lr.mean()), "sd_log_ratio": sd,
               "sd_log_beta_fit": comp["beta_fit"], "sd_log_beta_pred": comp["beta_pred"]}
        for m in (0.10, 0.25):
            mm = np.log(1 + m)
            z = stats.norm.ppf(0.95)
            row[f"tost_power_{int(m * 100)}"] = float(max(0.0, stats.norm.cdf((mm - 0) / sd - z) * 2 - 1))
        rows.append(row)
    # pooled constant: SD of the log ratio = kappa / sqrt(trials), least squares on log SD with slope -1/2
    tr = np.array([r["trials"] for r in rows], float)
    sd = np.array([r["sd_log_ratio"] for r in rows])
    kappa = float(np.exp(np.mean(np.log(sd) + 0.5 * np.log(tr))))
    z = stats.norm.ppf(0.95)
    need = {}
    for m in (0.10, 0.25):
        target = np.log(1 + m) / (z + stats.norm.ppf(0.90))      # TOST power 0.8 at true ratio 1: 2 Phi(m/sd - z) - 1 = 0.8
        need[f"trials_for_power80_{int(m * 100)}"] = float((kappa / target) ** 2)
    for r in rows:
        r["sd_pooled"] = kappa / np.sqrt(r["trials"])
        for m in (0.10, 0.25):
            r[f"pooled_power_{int(m * 100)}"] = float(max(0.0, 2 * stats.norm.cdf(np.log(1 + m) / r["sd_pooled"] - z) - 1))
    return {"rows": rows, "kappa": kappa, **need, "nrep": nrep}


def sensitivity(nrep=32, seed0=3000):
    out = {"tnd": {}, "neurons": {}, "exact_vs_gaussian": {}}
    tv = {0.25: [], 0.30: [], 0.35: []}
    for i in range(nrep):
        data = run_pipeline("integrator", seed0 + i, exact=False)
        for t in tv:
            tv[t].append(estimate(data, DESIGN, tnd_hat=t)["log_ratio"])
    out["tnd"] = {str(t): {"mean_log_ratio": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(nrep))} for t, v in tv.items()}
    for N in (50, 400):
        v = [estimate(run_pipeline("integrator", seed0 + 100 + i, exact=False, N=N), DESIGN)["log_ratio"] for i in range(nrep)]
        out["neurons"][str(N)] = {"mean_log_ratio": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                 "se": float(np.std(v, ddof=1) / np.sqrt(nrep))}
    for mode in ("exact", "gaussian"):
        v = [estimate(run_pipeline("integrator", seed0 + 200 + i, exact=(mode == "exact")), DESIGN)["log_ratio"]
             for i in range(16)]
        out["exact_vs_gaussian"][mode] = {"mean_log_ratio": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
    return out


def example_run(seed=7):
    """One run with exact Poisson counts: the stopped split-half cross moment, in units of the bound, against the
    stopped time, and all estimates."""
    data = run_pipeline("integrator", seed, exact=True)
    r = estimate(data, DESIGN)
    d, bw = DESIGN, DESIGN["bin"]
    z = data["zero"]
    sac = z["tau"] + d["tnd"]
    th = sac - d["tnd"]
    cap = np.where(np.isfinite(th), th, d["T"])
    nb = z["A"].shape[1]
    S1, S2 = stopped(z["A1"], th, sac, bw), stopped(z["A2"], th, sac, bw)
    st, pre = stop_time(cap, nb, bw)
    mom = np.mean((S1 - S1[:, :1]) * (S2 - S2[:, :1]), axis=0) / (r["B1"] * r["B2"])
    # the naive single-decoder increment variance and the survivors' variogram, same units
    tc = (np.arange(nb) + 0.5) * bw
    keep = tc <= 1.0
    return {"estimates": {k: r[k] for k in KEYS}, "tc": tc[keep], "stime": (st - pre * bw / 6).mean(0)[keep],
            "moment": mom[keep], "D_over_B2": r["D_over_B2"], "frac_decided_by_0.6": float(np.mean(th <= 0.6))}


def fp_curves(seed=4, n=6000):
    """Psychometric functions for parameter sets with the same D / (M B) and one with a different value."""
    rng = np.random.default_rng(seed)
    hs = np.linspace(-0.3, 0.3, 13)
    sets = [("D 0.8, M 8, B 1", 0.8, 8.0, 1.0), ("D 3.2, M 16, B 2", 3.2, 16.0, 2.0), ("D 0.2, M 4, B 0.5", 0.2, 4.0, 0.5),
            ("D 0.8, M 16, B 1", 0.8, 16.0, 1.0)]
    out = []
    for name, D, M, B in sets:
        pm = [float(np.mean(simulate_accumulator(h, n, rng, D=D, B=B, M=M, T=20.0)[0] > 0)) for h in hs]
        out.append({"name": name, "D": D, "M": M, "B": B, "theta": D / (M * B), "p_mc": pm,
                    "p_exact": [p_plus(M * h, D, B) for h in hs]})
    return {"h": hs, "sets": out, "n": n}


def departure_curves():
    Lam = np.linspace(-3, 6, 91)
    s = np.linspace(0, 3, 61)
    zz = np.linspace(0, 0.9, 46)
    return {"Lam": Lam, "leak": [leak_factor(x) for x in Lam], "s": s, "drift": [drift_var_factor(x) for x in s],
            "z": zz, "start_bias": [start_bias_factor(x) for x in zz], "start_var": [start_var_factor(x) for x in zz]}


def run():
    t0 = time.time()
    th = _cached("theory", theory_checks)
    cb = _cached("collapse", collapsing_bounds)
    fpc = _cached("fp_curves", fp_curves)
    dep = departure_curves()
    fdr = fdr_linear()
    ex = _cached("example", example_run)
    sct = _cached("scenarios", scenario_table)
    pw = _cached("power", power_analysis)
    sens = _cached("sensitivity", sensitivity)
    integ = sct["integrator"]
    tol = 3.5
    pure = cb[0]
    checks = {
        "exit_probabilities_match_closed_form": th["ddm"]["max_abs_zscore"] < tol,
        "rescaling_leaves_theta_and_choice_unchanged": abs(th["invariance"]["theta_rescaled"] - th["invariance"]["theta"]) < 1e-12
        and abs(th["invariance"]["p_rescaled"] - th["invariance"]["p"]) < 1e-12,
        "leak_factor_matches_quadrature": th["leak"]["max_err"] < 1e-5,
        "leak_exit_probabilities_match_mc": max(abs(r["zscore"]) for r in th["leak"]["mc"]) < tol,
        "drift_variability_matches_mc": abs(th["drift_var"]["mc"]["zscore"]) < tol,
        "start_variability_matches_mc": abs(th["start_var"]["mc"]["zscore"]) < tol,
        "start_bias_matches_mc": abs(th["start_bias"]["mc"]["zscore"]) < tol,
        "start_bias_factor_matches_numeric": max(abs(r["factor"] - r["factor_numeric"]) for r in th["start_bias"]["rows"]) < 1e-5,
        "pure_integrator_logistic_fit_unbiased": abs(pure["slope_fit"] - pure["naive_B0"]) < tol * pure["se"],
        "equal_theta_gives_equal_choice": max(abs(a - b) for st in fpc["sets"][:3] for a, b in zip(st["p_exact"], fpc["sets"][0]["p_exact"])) < 1e-12,
        "fluctuation_response_constant_for_ou": fdr["ou"]["max_rel_dev"] < 1e-10,
        "fisher_readout_identity": fdr["identity_residual"] < 1e-10
        and all(fdr["two_d"][c]["readouts"]["Fisher discriminant"]["max_rel_dev"] < 1e-10 for c in fdr["two_d"]),
        "fisher_equals_conjugate_at_detailed_balance": fdr["two_d"]["detailed balance"]["angle_deg"] < 1e-3,
        "other_readout_varies_at_detailed_balance": fdr["two_d"]["detailed balance"]["readouts"]["first coordinate"]["max_rel_dev"] > 0.05,
        "stopped_estimators_unbiased_for_integrator": abs(integ["log_ratio"]["mean"]) < max(0.03, 3 * integ["log_ratio"]["se"]),
        "naive_increment_estimator_fails": integ["beta_pred_naive"]["mean"] < 0.5 * integ["beta_fit"]["mean"],
        "survivor_variogram_biased_up": integ["beta_pred_vario"]["mean"] > 1.1 * integ["beta_fit"]["mean"],
        "power_improves_with_trials": pw["rows"][0]["sd_log_ratio"] > pw["rows"][-1]["sd_log_ratio"]
        and pw["rows"][-1]["tost_power_25"] > pw["rows"][0]["tost_power_25"],
        "gaussian_approximation_matches_poisson": abs(sens["exact_vs_gaussian"]["exact"]["mean_log_ratio"]
                                                       - sens["exact_vs_gaussian"]["gaussian"]["mean_log_ratio"])
        < 3 * np.hypot(sens["exact_vs_gaussian"]["exact"]["sd"], sens["exact_vs_gaussian"]["gaussian"]["sd"]) / 4,
    }
    return {"reference": REF, "design": DESIGN, "theory": th, "collapsing_bounds": cb, "fp_curves": fpc, "departures": dep,
            "fdr": fdr, "example": ex, "scenarios": sct, "power": pw, "sensitivity": sens,
            "checks": {k: bool(v) for k, v in checks.items()}, "seconds": time.time() - t0}


def _cached(name, fn):
    root = os.environ.get("FLUCT_CACHE")
    if not root:
        return fn()
    os.makedirs(root, exist_ok=True)
    p = os.path.join(root, name + ".pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    v = fn()
    pickle.dump(v, open(p, "wb"))
    return v
