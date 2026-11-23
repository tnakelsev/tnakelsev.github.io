"""
CADRE simulation: crop / livestock TFP disaggregation when shared inputs are
allocated between the two activities but the allocation is not observed.

The script builds an artificial farm economy in which the truth is known,
hides the allocation of labour and capital from the analyst, and compares
several ways of recovering crop TFP and livestock TFP growth:

  oracle     true allocation (not available in practice; the benchmark)
  revenue    shared inputs split in proportion to revenue (common practice)
  biophys    split by engineering requirement coefficients only
  behav      shadow-value rule, elasticity ratio taken from specialised farms
  cadre      triangulation: behavioural signal + biophysical prior, calibrated
             on a small validation subsample where the allocation is measured

It also checks the two accounting results of the note (Domar invariance of the
aggregate, opposite-sign bias from ignoring manure and residue flows) and
computes the range of sector TFP growth that the data allow without any
allocation assumption.

Run:  python3 cadre_sim.py            (writes CSVs and a figure to output/)
Only numpy, scipy, pandas and matplotlib are needed. Seeds are fixed.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "output", "disaggregation"))
# inside the project repository results go to output/disaggregation; as a
# stand-alone download they go to a cadre_output folder next to the script
OUT = os.environ.get("CADRE_OUT") or (_REPO_OUT if os.path.isdir(_REPO_OUT) else os.path.join(HERE, "cadre_output"))

# ----------------------------------------------------------------------------
# 1. Parameters of the artificial economy
# ----------------------------------------------------------------------------


@dataclass
class Par:
    N: int = 2000            # farm households
    T: int = 10              # years after the base year (T+1 observations)
    share_crop_only: float = 0.15
    share_liv_only: float = 0.15
    # output elasticities (constant returns in each activity)
    bLC: float = 0.35        # labour, crops
    bKC: float = 0.10        # capital, crops
    gC: float = 0.40         # land, crops
    nC: float = 0.15         # nutrients (fertiliser + manure), crops
    bLL: float = 0.20        # labour, livestock
    bKL: float = 0.15        # capital, livestock
    gL: float = 0.40         # herd, livestock
    fL: float = 0.25         # feed (purchased + crop residues), livestock
    # true TFP growth per year (log points)
    gA_C: float = 0.015
    gA_L: float = 0.005
    sd_year_C: float = 0.02  # weather-type common shocks
    sd_year_L: float = 0.01
    # endowment growth per year
    g_lab: float = 0.02
    g_cap: float = 0.03
    g_land: float = 0.01
    g_herd: float = 0.03
    g_PL: float = 0.02       # livestock price rises relative to crops
    a0_L: float = 0.05       # level of livestock technology (balances the two revenues)
    # by-products, valued at replacement cost
    res_value: float = 0.05  # residues are worth 5% of the grain value (base year)
    man_value: float = 0.03  # manure is worth 3% of the animal-output value (base year)
    g_integ: float = 0.0     # yearly growth of by-product recovery (scenario C)
    # allocation wedges (departures from the shadow-value rule)
    sd_tau_perm: float = 0.40
    sd_tau_iid: float = 0.20
    sd_eps: float = 0.20     # idiosyncratic output shocks
    # information available to the analyst
    val_frac: float = 0.10   # share of mixed farms with measured allocation
    sd_val_noise: float = 0.30
    prior_bias_lab: float = 1.4   # biophysical coefficients are wrong by these
    prior_bias_cap: float = 0.7   # factors (in odds terms)
    # scenario B: specialised crop farms use a different technology
    spec_diff: bool = False
    bLC_spec: float = 0.25
    gC_spec: float = 0.50


# ----------------------------------------------------------------------------
# 2. Data-generating process
# ----------------------------------------------------------------------------


def simulate(p: Par, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    N, T1 = p.N, p.T + 1
    u = rng.random(N)
    typ = np.where(u < p.share_crop_only, 1, np.where(u < p.share_crop_only + p.share_liv_only, 2, 0))
    mixed, conly, lonly = typ == 0, typ == 1, typ == 2
    has_c, has_l = ~lonly, ~conly

    def panel(mu, sd, g):
        base = rng.normal(mu, sd, N)[:, None]
        return np.exp(base + g * np.arange(T1)[None, :] + rng.normal(0, 0.10, (N, T1)))

    Lb = panel(np.log(300), 0.5, p.g_lab)
    K = panel(np.log(100), 0.7, p.g_cap)
    A = panel(np.log(4), 0.6, p.g_land) * has_c[:, None]
    H = panel(np.log(8), 0.8, p.g_herd) * has_l[:, None]

    z = rng.normal(0, 1, (N, 2))
    omC = 0.3 * z[:, 0]
    omL = 0.3 * (0.3 * z[:, 0] + np.sqrt(1 - 0.09) * z[:, 1])
    aC = p.gA_C * np.arange(T1) + rng.normal(0, p.sd_year_C, T1)
    aL = p.a0_L + p.gA_L * np.arange(T1) + rng.normal(0, p.sd_year_L, T1)

    bLC = np.where(conly & p.spec_diff, p.bLC_spec, p.bLC)
    gC = np.where(conly & p.spec_diff, p.gC_spec, p.gC)

    PC = np.ones(T1)
    PL = np.exp(p.g_PL * np.arange(T1))
    pF = np.ones(T1)      # fertiliser price (nutrient-equivalent unit)
    pPf = np.ones(T1)     # purchased feed price (feed-energy-equivalent unit)
    phi = p.res_value * PC[0] / pPf[0] * np.exp(p.g_integ * np.arange(T1))   # residue units per unit of crop output
    m = p.man_value * PL[0] / pF[0] * np.exp(p.g_integ * np.arange(T1))   # manure units per unit of animal output
    # effective output prices on mixed farms include the by-product
    PCe = np.where(mixed[:, None], (PC + pPf * phi)[None, :], PC[None, :])
    PLe = np.where(mixed[:, None], (PL + pF * m)[None, :], PL[None, :])

    tau_perm = rng.normal(0, p.sd_tau_perm, (N, 2))
    rho = np.array([p.bLC / p.bLL, p.bKC / p.bKL])

    names = ["thL", "thK", "YC", "YL", "F", "Mn", "Pf", "Rs"]
    S = {k: np.zeros((N, T1)) for k in names}

    def solve_year(t, Mn, Rs, tau, epsC, epsL):
        thL = np.where(conly, 1.0, np.where(lonly, 0.0, 0.5))
        thK = thL.copy()
        tiny = 1e-300
        for _ in range(300):
            BC = np.exp(aC[t] + omC) * np.maximum(thL * Lb[:, t], tiny) ** bLC \
                * np.maximum(thK * K[:, t], tiny) ** p.bKC * np.maximum(A[:, t], tiny) ** gC
            Nn = np.maximum((p.nC * PCe[:, t] * BC / pF[t]) ** (1 / (1 - p.nC)), Mn)
            EYC = BC * Nn ** p.nC * has_c
            BL = np.exp(aL[t] + omL) * np.maximum((1 - thL) * Lb[:, t], tiny) ** p.bLL \
                * np.maximum((1 - thK) * K[:, t], tiny) ** p.bKL * np.maximum(H[:, t], tiny) ** p.gL
            Fd = np.maximum((p.fL * PLe[:, t] * BL / pPf[t]) ** (1 / (1 - p.fL)), Rs)
            EYL = BL * Fd ** p.fL * has_l
            ratio = (PCe[:, t] * EYC + tiny) / (PLe[:, t] * EYL + tiny)
            oL = rho[0] * ratio * np.exp(tau[:, 0])
            oK = rho[1] * ratio * np.exp(tau[:, 1])
            nL = np.where(mixed, oL / (1 + oL), thL)
            nK = np.where(mixed, oK / (1 + oK), thK)
            d = max(np.abs(nL - thL).max(), np.abs(nK - thK).max())
            thL, thK = 0.5 * thL + 0.5 * nL, 0.5 * thK + 0.5 * nK
            if d < 1e-10:
                break
        YC = EYC * np.exp(epsC)
        YL = EYL * np.exp(epsL)
        return thL, thK, YC, YL, (Nn - Mn) * has_c, Mn, (Fd - Rs) * has_l, Rs

    # burn-in: iterate the base year so that lagged flows are consistent
    Mn = np.zeros(N)
    Rs = np.zeros(N)
    tau0 = tau_perm.copy()
    for _ in range(6):
        r = solve_year(0, Mn, Rs, tau0, 0.0, 0.0)
        Mn = m[0] * r[3] * mixed
        Rs = phi[0] * r[2] * mixed
    for t in range(T1):
        tau = tau_perm + rng.normal(0, p.sd_tau_iid, (N, 2))
        r = solve_year(t, Mn, Rs, tau, rng.normal(0, p.sd_eps, N), rng.normal(0, p.sd_eps, N))
        for k, v in zip(names, r):
            S[k][:, t] = v
        Mn = m[min(t + 1, p.T)] * r[3] * mixed     # manure available next year
        Rs = phi[min(t + 1, p.T)] * r[2] * mixed   # residues available next year

    RC = PCe * S["YC"]
    RL = PLe * S["YL"]
    # factor prices observed by the analyst (national averages)
    wL = ((bLC[:, None] * RC + p.bLL * RL).sum(0)) / Lb.sum(0)
    wK = ((p.bKC * RC + p.bKL * RL).sum(0)) / K.sum(0)
    rA = (gC[:, None] * RC).sum(0) / A.sum(0)
    rH = (p.gL * RL).sum(0) / H.sum(0)

    # validation subsample and biophysical prior
    val = mixed & (rng.random(N) < p.val_frac)
    noise = rng.normal(0, p.sd_val_noise, (N, T1, 2))
    logit = lambda x: np.log(x / (1 - x))
    with np.errstate(divide="ignore", invalid="ignore"):
        obsL = logit(S["thL"]) + noise[:, :, 0]
        obsK = logit(S["thK"]) + noise[:, :, 1]
        AH = np.log(A / H)
        # engineering coefficients: right on average in the base year, then
        # deliberately distorted by the bias factors
        cL = np.median((logit(S["thL"]) - AH)[mixed, 0]) + np.log(p.prior_bias_lab)
        cK = np.median((logit(S["thK"]) - AH)[mixed, 0]) + np.log(p.prior_bias_cap)
        priorL, priorK = cL + AH, cK + AH   # prior log-odds

    return dict(p=p, mixed=mixed, conly=conly, lonly=lonly, Lb=Lb, K=K, A=A, H=H, **S,
                RC=RC, RL=RL, PC=PC, PL=PL, pF=pF, pPf=pPf, phi=phi, m=m,
                wL=wL, wK=wK, rA=rA, rH=rH, val=val, obsL=obsL, obsK=obsK,
                priorL=priorL, priorK=priorK, aC=aC, aL=aL)


# ----------------------------------------------------------------------------
# 3. Index numbers (Tornqvist, revenue-based weights)
# ----------------------------------------------------------------------------


def _wsum(items, denom):
    """Tornqvist-weighted sum of log changes; items = [(value, quantity), ...]."""
    g = 0.0
    for val, q in items:
        s = val / denom
        g = g + 0.5 * (s[1:] + s[:-1]) * np.diff(np.log(q))
    return g


def sector_tfp(outputs, inputs):
    R = sum(v for v, _ in outputs)
    return _wsum(outputs, R) - _wsum(inputs, R), R


def accounts(d, xLC, xKC, flows=True):
    """Sector and aggregate TFP growth given national crop allocations xLC, xKC."""
    mx = d["mixed"]
    LbT, KT = d["Lb"].sum(0), d["K"].sum(0)
    YC, YL = d["YC"].sum(0), d["YL"].sum(0)
    A, H, F, Pf = d["A"].sum(0), d["H"].sum(0), d["F"].sum(0), d["Pf"].sum(0)
    Mn, Rs = d["Mn"].sum(0), d["Rs"].sum(0)          # flows used this year
    resid = d["phi"] * d["YC"][mx].sum(0)            # by-products produced
    manure = d["m"] * d["YL"][mx].sum(0)
    outC = [(d["PC"] * YC, YC)]
    outL = [(d["PL"] * YL, YL)]
    inC = [(d["wL"] * xLC, xLC), (d["wK"] * xKC, xKC), (d["rA"] * A, A), (d["pF"] * F, F)]
    inL = [(d["wL"] * (LbT - xLC), LbT - xLC), (d["wK"] * (KT - xKC), KT - xKC),
           (d["rH"] * H, H), (d["pPf"] * Pf, Pf)]
    intra = []
    if flows:
        outC.append((d["pPf"] * resid, resid))
        outL.append((d["pF"] * manure, manure))
        inC.append((d["pF"] * Mn, Mn))
        inL.append((d["pPf"] * Rs, Rs))
        intra = [(d["pF"] * Mn, Mn), (d["pPf"] * Rs, Rs)]
    gC, RC = sector_tfp(outC, inC)
    gL, RL = sector_tfp(outL, inL)
    V = RC + RL - sum(v for v, _ in intra)
    prim = [(d["wL"] * LbT, LbT), (d["wK"] * KT, KT), (d["rA"] * A, A), (d["rH"] * H, H),
            (d["pF"] * F, F), (d["pPf"] * Pf, Pf)]
    gAgg = _wsum(outC + outL, V) - _wsum(intra, V) - _wsum(prim, V)
    dC, dL = RC / V, RL / V
    domar = 0.5 * (dC[1:] + dC[:-1]) * gC + 0.5 * (dL[1:] + dL[:-1]) * gL
    return dict(gC=gC, gL=gL, gAgg=gAgg, domar=domar, dC=dC, dL=dL)


# ----------------------------------------------------------------------------
# 4. Allocation estimators
# ----------------------------------------------------------------------------

expit = lambda x: 1 / (1 + np.exp(-x))


def national(d, thL, thK):
    """Add specialised farms (allocation known) to estimated shares of mixed farms."""
    mx, co = d["mixed"], d["conly"]
    xL = (thL[mx] * d["Lb"][mx]).sum(0) + d["Lb"][co].sum(0)
    xK = (thK[mx] * d["K"][mx]).sum(0) + d["K"][co].sum(0)
    return xL, xK


def rho_from_specialised(d):
    """Elasticity ratio (crops / livestock) for labour and capital, estimated on
    specialised farms. The flexible input (fertiliser, feed) is removed with its
    revenue share so that its endogeneity does not contaminate the regression."""
    T1 = d["Lb"].shape[1]

    def fit(mask, Y, R, flex, pflex, fixed):
        sh = np.median((pflex[None, :] * flex[mask]) / R[mask])
        y = (np.log(Y[mask]) - sh * np.log(flex[mask])).ravel()
        X = [np.log(d["Lb"][mask]).ravel(), np.log(d["K"][mask]).ravel(), np.log(fixed[mask]).ravel()]
        X = np.column_stack(X + [np.tile(np.eye(T1), (mask.sum(), 1))])
        XtXi = np.linalg.inv(X.T @ X)
        b = XtXi @ X.T @ y
        G, k = mask.sum(), X.shape[1]
        sc = np.einsum("gtk,gt->gk", X.reshape(G, T1, k), (y - X @ b).reshape(G, T1))
        V = XtXi @ (sc.T @ sc) @ XtXi * G / (G - 1)      # clustered by household
        return b[:2], np.diag(V)[:2]

    bC, vC = fit(d["conly"], d["YC"], d["RC"], d["F"], d["pF"], d["A"])
    bL, vL = fit(d["lonly"], d["YL"], d["RL"], d["Pf"], d["pPf"], d["H"])
    se_log = np.sqrt(vC / bC ** 2 + vL / bL ** 2)       # delta method
    return bC / bL, se_log


def estimate(d, rng=None, draws=0):
    """Return a dict of national crop allocations (xLC, xKC) by method."""
    mx = d["mixed"]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(d["RC"] / d["RL"])
    out = {}
    out["oracle"] = national(d, d["thL"], d["thK"])
    th = expit(lr)
    out["revenue"] = national(d, th, th)
    out["biophys"] = national(d, expit(d["priorL"]), expit(d["priorK"]))
    rho, se_rho = rho_from_specialised(d)
    out["behav"] = national(d, expit(np.log(rho[0]) + lr), expit(np.log(rho[1]) + lr))

    # CADRE: calibration regression on the validation subsample
    v = d["val"]
    info = {"rho_spec": rho}
    th_hat, sims = [], []
    for j, (obs, prior) in enumerate([(d["obsL"], d["priorL"]), (d["obsK"], d["priorK"])]):
        X = np.column_stack([np.ones(v.sum() * lr.shape[1]), lr[v].ravel(), prior[v].ravel()])
        y = obs[v].ravel()
        XtXi = np.linalg.inv(X.T @ X)
        b = XtXi @ X.T @ y
        e = y - X @ b
        s2 = e @ e / (len(y) - 3)
        # households are clusters: use a cluster-robust covariance
        G = v.sum()
        Xg = X.reshape(G, -1, 3)
        eg = e.reshape(G, -1)
        meat = sum(np.outer(Xg[g].T @ eg[g], Xg[g].T @ eg[g]) for g in range(G))
        Vb = XtXi @ meat @ XtXi * G / (G - 1)
        with np.errstate(invalid="ignore"):
            pred = b[0] + b[1] * lr + b[2] * prior
        th_hat.append(expit(pred))
        # over-identification check: is the level of the observed log-odds what
        # the specialised-farm technology predicts?
        dev = (obs[v] - lr[v]).mean(1) - np.log(rho[j])
        info[f"coef_{'LK'[j]}"] = b
        info[f"t_spec_{'LK'[j]}"] = dev.mean() / np.sqrt(dev.var(ddof=1) / G + se_rho[j] ** 2)
        if draws:
            sj = []
            for _ in range(draws):
                bd = rng.multivariate_normal(b, Vb)
                with np.errstate(invalid="ignore"):
                    sj.append(expit(bd[0] + bd[1] * lr + bd[2] * prior + rng.normal(0, np.sqrt(s2), lr.shape)))
            sims.append(sj)
    out["cadre"] = national(d, th_hat[0], th_hat[1])
    if draws:
        info["draws"] = [national(d, sims[0][r], sims[1][r]) for r in range(draws)]
    return out, info


# ----------------------------------------------------------------------------
# 5. Range allowed by the data without an allocation rule
# ----------------------------------------------------------------------------


def bounds(d, kappa=None):
    """Smallest and largest average crop / livestock TFP growth compatible with
    the data. kappa=None: only specialised farms restrict the allocation.
    kappa=k: each mixed farm's allocation odds lie within a factor k of the
    biophysical prior."""
    mx, co = d["mixed"], d["conly"]
    T1 = d["Lb"].shape[1]
    lo, hi = [], []
    for X, prior in [(d["Lb"], d["priorL"]), (d["K"], d["priorK"])]:
        base = X[co].sum(0)
        if kappa is None:
            lo.append(base + 1e-6 * X[mx].sum(0))
            hi.append(base + (1 - 1e-6) * X[mx].sum(0))
        else:
            lo.append(base + (expit(prior[mx] - np.log(kappa)) * X[mx]).sum(0))
            hi.append(base + (expit(prior[mx] + np.log(kappa)) * X[mx]).sum(0))
    lo, hi = np.concatenate(lo), np.concatenate(hi)

    def obj(u, key, sign):
        x = lo + u * (hi - lo)
        a = accounts(d, x[:T1], x[T1:])
        return sign * a[key].mean()

    res = {}
    ramp = np.linspace(0, 1, T1)
    starts = [np.tile(ramp, 2), np.tile(ramp[::-1], 2), np.full(2 * T1, 0.5)]
    for key in ("gC", "gL"):
        for sign, lab in ((1, "min"), (-1, "max")):
            best = None
            for s in starts:
                r = minimize(obj, s, args=(key, sign), method="L-BFGS-B", bounds=[(0, 1)] * (2 * T1))
                if best is None or r.fun < best.fun:
                    best = r
            x = lo + best.x * (hi - lo)
            a = accounts(d, x[:T1], x[T1:])
            res[(key, lab)] = (a["gC"].mean(), a["gL"].mean())
    return res, (lo, hi)


# ----------------------------------------------------------------------------
# 6. Experiments
# ----------------------------------------------------------------------------

METHODS = ["oracle", "revenue", "biophys", "behav", "cadre"]


def one_rep(p, seed):
    d = simulate(p, seed)
    alloc, info = estimate(d)
    row = {}
    for mth in METHODS:
        a = accounts(d, *alloc[mth])
        row[f"{mth}_C"] = 100 * a["gC"].mean()
        row[f"{mth}_L"] = 100 * a["gL"].mean()
        row[f"{mth}_agg"] = 100 * a["gAgg"].mean()
        row[f"{mth}_domar"] = 100 * a["domar"].mean()
        row[f"{mth}_shL"] = 100 * (alloc[mth][0] / d["Lb"].sum(0)).mean()
        row[f"{mth}_shK"] = 100 * (alloc[mth][1] / d["K"].sum(0)).mean()
    a = accounts(d, *alloc["oracle"], flows=False)
    row["noflow_C"], row["noflow_L"], row["noflow_agg"] = (100 * a[k].mean() for k in ("gC", "gL", "gAgg"))
    row["struct_C"] = 100 * (d["aC"][-1] - d["aC"][0]) / p.T
    row["struct_L"] = 100 * (d["aL"][-1] - d["aL"][0]) / p.T
    row["t_spec_L"], row["t_spec_K"] = info["t_spec_L"], info["t_spec_K"]
    return row


def monte_carlo(p, reps, seed0=1000):
    return pd.DataFrame([one_rep(p, seed0 + r) for r in range(reps)])


def summarise(df, label):
    rows = []
    for mth in METHODS[1:] + ["noflow"]:
        for k, name in (("C", "crops"), ("L", "livestock")):
            err = df[f"{mth}_{k}"] - df[f"oracle_{k}"]
            rows.append(dict(scenario=label, method=mth, sector=name,
                             true_growth=df[f"oracle_{k}"].mean(), estimate=df[f"{mth}_{k}"].mean(),
                             bias=err.mean(), rmse=np.sqrt((err ** 2).mean()),
                             worst_abs_error=err.abs().max()))
    for mth in METHODS[1:]:
        for k, name in (("shL", "labour share to crops (%)"), ("shK", "capital share to crops (%)")):
            err = df[f"{mth}_{k}"] - df[f"oracle_{k}"]
            rows.append(dict(scenario=label, method=mth, sector=name,
                             true_growth=df[f"oracle_{k}"].mean(), estimate=df[f"{mth}_{k}"].mean(),
                             bias=err.mean(), rmse=np.sqrt((err ** 2).mean()),
                             worst_abs_error=err.abs().max()))
    return pd.DataFrame(rows)


def main(reps=200):
    os.makedirs(OUT, exist_ok=True)
    pA, pB, pC = Par(), Par(spec_diff=True), Par(g_integ=0.08)
    mcA, mcB, mcC = monte_carlo(pA, reps), monte_carlo(pB, reps), monte_carlo(pC, reps)
    mcC.to_csv(os.path.join(OUT, "mc_replications_scenarioC.csv"), index=False)
    mcA.to_csv(os.path.join(OUT, "mc_replications_scenarioA.csv"), index=False)
    mcB.to_csv(os.path.join(OUT, "mc_replications_scenarioB.csv"), index=False)
    summ = pd.concat([summarise(mcA, "A: same technology on specialised farms"),
                      summarise(mcB, "B: specialised crop farms differ"),
                      summarise(mcC, "C: crop-livestock integration deepens")])
    summ.to_csv(os.path.join(OUT, "T1_method_comparison.csv"), index=False, float_format="%.4f")

    # accounting checks
    chk = []
    for lab, mc in (("A", mcA), ("B", mcB), ("C", mcC)):
        aggs = mc[[f"{m_}_agg" for m_ in METHODS]]
        chk.append(dict(scenario=lab,
                        oracle_structural_gap_C=(mc["oracle_C"] - mc["struct_C"]).abs().mean(),
                        oracle_structural_gap_L=(mc["oracle_L"] - mc["struct_L"]).abs().mean(),
                        max_spread_aggregate_across_methods=(aggs.max(axis=1) - aggs.min(axis=1)).max(),
                        max_abs_domar_gap=max((mc[f"{m_}_domar"] - mc[f"{m_}_agg"]).abs().max() for m_ in METHODS),
                        noflow_shift_aggregate=(mc["noflow_agg"] - mc["oracle_agg"]).abs().max(),
                        reject_share_spec_test_L=(mc["t_spec_L"].abs() > 1.96).mean(),
                        reject_share_spec_test_K=(mc["t_spec_K"].abs() > 1.96).mean()))
    pd.DataFrame(chk).to_csv(os.path.join(OUT, "T2_accounting_checks.csv"), index=False, float_format="%.5f")

    # one illustrative economy: intervals and identified ranges
    d = simulate(pA, 7)
    rng = np.random.default_rng(77)
    alloc, info = estimate(d, rng=rng, draws=200)
    pts = {m_: accounts(d, *alloc[m_]) for m_ in METHODS}
    sims = np.array([[100 * accounts(d, *x)[k].mean() for k in ("gC", "gL")] for x in info["draws"]])
    b_none, _ = bounds(d, None)
    b_bio, _ = bounds(d, 3.0)
    ill = []
    for m_ in METHODS:
        ill.append(dict(item=m_, crop=100 * pts[m_]["gC"].mean(), livestock=100 * pts[m_]["gL"].mean()))
    ill.append(dict(item="cadre_2.5pct", crop=np.percentile(sims[:, 0], 2.5), livestock=np.percentile(sims[:, 1], 2.5)))
    ill.append(dict(item="cadre_97.5pct", crop=np.percentile(sims[:, 0], 97.5), livestock=np.percentile(sims[:, 1], 97.5)))
    for lab, b in (("range_no_assumption", b_none), ("range_biophysical_x3", b_bio)):
        ill.append(dict(item=lab + "_min", crop=100 * b[("gC", "min")][0], livestock=100 * b[("gL", "min")][1]))
        ill.append(dict(item=lab + "_max", crop=100 * b[("gC", "max")][0], livestock=100 * b[("gL", "max")][1]))
    ill = pd.DataFrame(ill)
    ill.to_csv(os.path.join(OUT, "T3_illustrative_economy.csv"), index=False, float_format="%.4f")
    coefs = pd.DataFrame({"labour": info["coef_L"], "capital": info["coef_K"]},
                         index=["constant", "log revenue ratio", "biophysical prior log-odds"])
    coefs.loc["log elasticity ratio from specialised farms"] = np.log(info["rho_spec"])
    coefs.to_csv(os.path.join(OUT, "T4_calibration_coefficients.csv"), float_format="%.4f")
    return summ, pd.DataFrame(chk), ill, coefs, dict(d=d, pts=pts, b_none=b_none, b_bio=b_bio, sims=sims)


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    summ, chk, ill, coefs, _ = main(reps)
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    print(chk.T.to_string())
    print(ill.to_string(index=False))
    print(coefs.to_string())
