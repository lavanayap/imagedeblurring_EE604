"""
ADAPTIVE NOISE- AND PSF-UNCERTAINTY-AWARE RL-TV CONTROLLER (3D)
================================================================
Project: Noise-Adaptive and PSF-Robust Fluorescence Microscopy Deconvolution

THE CONTROLLER NEVER SEES THE REFERENCE VOLUME.
Its only inputs are the degraded volume `g` and the assumed PSF `psf`.

Three connected ideas (the proposed contribution):

1. NOISE-AWARENESS
   A Poisson-Gaussian variance model  var(I) = a*I + b  is estimated from
   the degraded volume alone (robust MAD of a 3D Laplacian residual, binned
   by local intensity, then non-negative least squares).

2. ADAPTIVE STOPPING + ADAPTIVE LAMBDA  (discrepancy principle)
   During RL-TV, the normalised residual
        chi2_k = mean( (g - H f_k)^2 / (a*H f_k + b) )
   is monitored. At the true solution chi2 ~ 1.  We stop at the first k with
   chi2_k <= tau.  Among a grid of TV weights lambda, we choose the LARGEST
   lambda whose run still reaches the discrepancy within n_max iterations
   (the most regularised solution that is still consistent with the noise).

3. PSF-UNCERTAINTY-AWARENESS  (consistency test)
   An assumed PSF that is TOO WIDE cannot explain the data: RL never reaches
   the noise level (chi2 stays > tau). So: if the assumed PSF (s=1) reaches
   the discrepancy it is kept. Otherwise narrower versions (s<1, PSF width
   rescaled) are tried from the closest to the furthest, and the first one
   consistent with the data is used. If none is, the assumed PSF is kept.
   Residual whiteness is logged as a diagnostic only.
   KNOWN ASYMMETRY: a PSF that is too NARROW is NOT detectable this way
   (a narrower PSF always fits easily), so only over-wide PSFs are corrected.

Ablation variants are produced from the same cached runs so that every
component can be switched off individually.

Limitations (state these in the report):
- Noise model is Poisson-Gaussian; clipping of negative values in the
  calibrated input biases the variance estimate at low intensity.
- The PSF stage only detects/corrects over-wide PSFs (see asymmetry above).
- Only an isotropic width scaling of the PSF is searched.
- Periodic (FFT) boundaries; denominator clip 0.5..2 is a stability guard.
"""

import numpy as np
from scipy.fft import rfftn, irfftn, ifftshift
from scipy.ndimage import affine_transform
from scipy.optimize import nnls

WORKERS = -1
EPS = 1e-8
EPS_TV = 1e-3


# =====================================================
# BASIC OPERATORS (same maths as rl_tv/rl_tv_final.py)
# =====================================================

def normalize_psf(psf):
    psf = np.maximum(np.asarray(psf, dtype=np.float32), 0)
    total = float(np.sum(psf, dtype=np.float64))
    if total <= 0:
        raise ValueError("PSF has zero energy")
    return (psf / total).astype(np.float32)


def make_otf(psf, shape):
    return rfftn(ifftshift(normalize_psf(psf)), s=shape, workers=WORKERS)


def conv(x, H):
    return irfftn(rfftn(x, workers=WORKERS) * H, s=x.shape, workers=WORKERS)


def corr(x, H):
    return irfftn(rfftn(x, workers=WORKERS) * np.conj(H),
                  s=x.shape, workers=WORKERS)


def grad(u):
    return [np.roll(u, -1, axis=a) - u for a in range(u.ndim)]


def div(p):
    return sum(pa - np.roll(pa, 1, axis=a) for a, pa in enumerate(p))


def scale_psf(psf, s):
    """Rescale PSF width about its centre voxel by factor s (s=1 -> unchanged)."""
    psf = normalize_psf(psf)
    if abs(s - 1.0) < 1e-9:
        return psf
    centre = np.array([d // 2 for d in psf.shape], dtype=np.float64)
    out = affine_transform(psf, matrix=np.eye(psf.ndim) / s,
                           offset=centre * (1.0 - 1.0 / s),
                           order=1, mode="constant", cval=0.0)
    return normalize_psf(out)


# =====================================================
# 1. NOISE ESTIMATION (reference-free)
# =====================================================

SIGMA_FLOOR = 2e-4      # ~ a few 8/16-bit quantisation steps; avoids sigma -> 0


def estimate_noise_model(g, n_bins=12, min_count=2000):
    """
    Estimate Poisson-Gaussian variance  var = a*mu + b  from g alone.

    Residual: 3D 7-point Laplacian  L = sum(6 neighbours) - 6*centre.
    For white noise  var(L) = 42*sigma^2.  MAD makes it robust to the
    (smooth, blurred) signal.

    Only UNCLIPPED voxels (centre and all 6 neighbours > 0) are used. The
    calibrated input is clipped at 0, so large flat zero regions would
    otherwise drive the MAD (and sigma) to zero.

    Returns dict(a, b, sigma_global, n_bins_used); sigma_global = noise std
    at the dark end of the data.
    """
    g = np.asarray(g, dtype=np.float32)
    c = g[1:-1, 1:-1, 1:-1]
    nbrs = (g[:-2, 1:-1, 1:-1], g[2:, 1:-1, 1:-1],
            g[1:-1, :-2, 1:-1], g[1:-1, 2:, 1:-1],
            g[1:-1, 1:-1, :-2], g[1:-1, 1:-1, 2:])
    nb = sum(nbrs)
    valid = c > 0
    for n_ in nbrs:
        valid &= n_ > 0
    if int(valid.sum()) < min_count:             # almost everything clipped
        valid = np.ones_like(valid, dtype=bool)

    lap = (nb - 6.0 * c)[valid].astype(np.float64)
    mu = ((nb + c) / 7.0)[valid].astype(np.float64)

    def robust_var(x):
        mad = np.median(np.abs(x - np.median(x)))
        return (1.4826 * mad) ** 2 / 42.0

    sigma_all = float(max(np.sqrt(robust_var(lap)), SIGMA_FLOOR))

    edges = np.quantile(mu, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, mu, side="right") - 1, 0, n_bins - 1)

    mus, vars_, wts = [], [], []
    for bidx in range(n_bins):
        m = idx == bidx
        n = int(m.sum())
        if n < min_count:
            continue
        v = robust_var(lap[m])
        if v <= 0:
            continue
        mus.append(float(np.median(mu[m])))
        vars_.append(v)
        wts.append(np.sqrt(n))

    if len(mus) >= 2:
        mus, vars_, wts = map(np.asarray, (mus, vars_, wts))
        A = np.stack([mus, np.ones_like(mus)], axis=1) * wts[:, None]
        coef, _ = nnls(A, vars_ * wts)
        a, b = float(coef[0]), float(coef[1])
    else:                                        # fallback: homoscedastic
        a, b = 0.0, sigma_all ** 2

    b = max(b, SIGMA_FLOOR ** 2)
    # noise std at the DARK end of the valid voxels: used for the signal mask
    # and variance floor (a global MAD would be dominated by bright voxels)
    sigma_global = float(np.sqrt(max(a * np.quantile(mu, 0.05) + b,
                                     SIGMA_FLOOR ** 2)))
    return {"a": a, "b": b, "sigma_global": sigma_global,
            "n_bins_used": len(mus)}


def variance_model(pred, noise):
    floor = (0.5 * noise["sigma_global"]) ** 2
    return np.maximum(noise["a"] * pred + noise["b"], max(floor, 1e-10))


def signal_mask(g, noise, k=4.0, min_frac=0.01):
    """
    Voxels whose local mean is clearly above the noise floor (k*sigma).

    Calibrated data are clipped at 0, which distorts the noise statistics of
    dark background voxels. Discrepancy statistics are therefore computed on
    signal voxels only, where the Poisson-Gaussian model is trustworthy.
    """
    g = np.asarray(g, dtype=np.float32)
    acc = g.copy()
    for ax in range(g.ndim):
        acc += np.roll(g, 1, axis=ax) + np.roll(g, -1, axis=ax)
    mask = (acc / (1 + 2 * g.ndim)) > k * noise["sigma_global"]
    if mask.mean() < min_frac:
        mask = np.ones_like(mask, dtype=bool)
    return mask


# =====================================================
# 2. RESIDUAL DIAGNOSTICS
# =====================================================

def residual_stats(g, pred, noise, mask=None):
    """
    Return (chi2, whiteness) of the variance-normalised residual, evaluated
    on `mask` (signal voxels) if given.
    """
    r = (g - pred) / np.sqrt(variance_model(pred, noise))
    if mask is None:
        mask = np.ones(r.shape, dtype=bool)
    m = mask.astype(np.float32)
    n = float(m.sum()) + 1e-12
    rm = r * m
    chi2 = float(np.sum(rm.astype(np.float64) ** 2) / n)
    den = float(np.sum(rm.astype(np.float64) ** 2)) + 1e-12
    rho = [float(np.sum(rm * np.roll(rm, -1, axis=ax), dtype=np.float64)) / den
           for ax in range(r.ndim)]
    white = float(np.sqrt(np.mean(np.square(rho))))
    return chi2, white


# =====================================================
# 3. RL-TV WITH DISCREPANCY STOPPING
# =====================================================

def run_rl_tv(g, H, noise, lam, n_max=100, tau=1.0, snap_iters=(),
              plateau_tol=0.0, plateau_window=10, plateau_min_iter=50):
    """
    RL-TV (Dey et al. 2006) that monitors chi2 each iteration.

    Stopping: (1) chi2 <= tau  (discrepancy principle), else (2) if
    plateau_tol > 0: after plateau_min_iter iterations, when chi2 improved by
    less than plateau_tol*plateau_window (relative) over the last
    plateau_window iterations, else (3) n_max.

    Returns dict:
        volume      estimate at the first k with chi2<=tau (or at n_max)
        iterations  that k
        reached     True if the discrepancy (chi2<=tau) was reached
        stop_reason 'discrepancy' | 'plateau' | 'n_max'
        chi2, whiteness  at the returned iterate
        snaps       {k: volume} for k in snap_iters
        trace       per-iteration chi2 / whiteness
    """
    g = np.maximum(np.asarray(g, dtype=np.float32), EPS)
    f = np.full_like(g, g.mean())
    last_snap = max(snap_iters) if snap_iters else 0
    mask = signal_mask(g, noise)
    snaps, trace, chi_hist = {}, [], []
    stop = None

    for k in range(0, n_max + 1):
        pred = conv(f, H)
        chi2, white = residual_stats(g, pred, noise, mask)
        trace.append({"iteration": k, "chi2": chi2, "whiteness": white})

        if k in snap_iters:
            snaps[k] = f.copy()
        chi_hist.append(chi2)
        if stop is None and k >= 1 and chi2 <= tau:
            stop = {"volume": f.copy(), "iterations": k, "chi2": chi2,
                    "whiteness": white, "stop_reason": "discrepancy"}
        elif (stop is None and plateau_tol > 0
              and k >= max(plateau_min_iter, plateau_window)):
            ref = chi_hist[k - plateau_window]
            if ref - chi2 < plateau_tol * plateau_window * ref:
                stop = {"volume": f.copy(), "iterations": k, "chi2": chi2,
                        "whiteness": white, "stop_reason": "plateau"}
        if k == n_max or (stop is not None and k >= last_snap):
            break

        ratio = g / np.maximum(pred, EPS)
        data_term = corr(ratio, H)
        if lam > 0:
            gr = grad(f)
            norm = np.sqrt(sum(a ** 2 for a in gr) + EPS_TV ** 2)
            tv = div([a / norm for a in gr])
            denom = np.clip(1.0 - lam * tv, 0.5, 2.0)
            f = f * data_term / denom
        else:
            f = f * data_term
        f = np.maximum(f, 0).astype(np.float32)

    if stop is None:
        stop = {"volume": f, "iterations": n_max,
                "chi2": trace[-1]["chi2"], "whiteness": trace[-1]["whiteness"],
                "stop_reason": "n_max"}
    stop.update({"reached": stop["stop_reason"] == "discrepancy", "lam": lam,
                 "snaps": snaps, "trace": trace})
    return stop


# =====================================================
# 4. THE ADAPTIVE CONTROLLER
# =====================================================

class AdaptiveDeconvolver:
    """
    restore(g) -> (volume, info)      full adaptive method
    variants(g) -> {name: (volume, info)}   ablations (shares cached runs)

    Parameters
    ----------
    psf            assumed PSF, same shape as g (centred)
    scales         PSF width multipliers searched (must contain 1.0)
    lambdas        TV weights searched (must contain default_lambda)
    n_max          maximum RL iterations
    tau            discrepancy tolerance on chi2 (1.0 = expected noise level)
    default_lambda lambda used by the "no lambda adaptation" ablation
    fixed_iters    iteration count used by the "no adaptive stopping" ablation
    consist_tol    a PSF scale counts as data-consistent if its final chi2 is
                   <= tau*(1+consist_tol) (also when stopped by the plateau rule)
    plateau_tol    also stop when chi2 improves < plateau_tol per iteration
                   (0 disables); needs plateau_min_iter iterations first
    fit_tol        if NO lambda reaches the discrepancy (model mismatch), pick
                   the largest lambda whose chi2 is within (1+fit_tol) of the
                   best chi2 over the grid
    """

    def __init__(self, psf, scales=(0.7, 0.8, 0.9, 1.0),
                 lambdas=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1), n_max=200, tau=1.0,
                 default_lambda=1e-2, fixed_iters=50, fit_tol=0.02,
                 consist_tol=0.05, plateau_tol=0.001, plateau_window=10, plateau_min_iter=50,
                 log=print):
        if 1.0 not in scales:
            raise ValueError("scales must contain 1.0")
        if default_lambda not in lambdas:
            raise ValueError("lambdas must contain default_lambda")
        self.psf = normalize_psf(psf)
        self.scales = sorted(scales)
        self.lambdas = sorted(lambdas)
        self.n_max, self.tau = n_max, tau
        self.default_lambda, self.fixed_iters = default_lambda, fixed_iters
        self.fit_tol, self.consist_tol = fit_tol, consist_tol
        self.plateau = (plateau_tol, plateau_window, plateau_min_iter)
        self.log = log

    # ---- internals -------------------------------------------------
    def _reset(self, g):
        self.g = g
        self.noise = estimate_noise_model(g)
        self.cache = {}
        self.runs = []                      # flat log for CSV
        self._H = {}

    def _otf(self, s):
        if s not in self._H:
            self._H[s] = make_otf(scale_psf(self.psf, s), self.g.shape)
        return self._H[s]

    def _run(self, s, lam, snaps=()):
        key = (s, lam)
        if key not in self.cache:
            r = run_rl_tv(self.g, self._otf(s), self.noise, lam,
                          self.n_max, self.tau, snap_iters=snaps,
                          plateau_tol=self.plateau[0],
                          plateau_window=self.plateau[1],
                          plateau_min_iter=self.plateau[2])
            self.cache[key] = r
            self.runs.append({"scale": s, "lambda": lam,
                              "iterations": r["iterations"],
                              "reached": int(r["reached"]),
                              "stop_reason": r["stop_reason"],
                              "chi2": r["chi2"],
                              "whiteness": r["whiteness"]})
            self.log(f"   run s={s:<5g} lam={lam:<7g} -> it={r['iterations']:3d} "
                     f"stop={r['stop_reason']:<11s} chi2={r['chi2']:.3f} "
                     f"white={r['whiteness']:.4f}")
        return self.cache[key]

    def select_scale(self):
        """
        Stage 1: PSF consistency test (see module docstring).
        Keeps s=1 if consistent; else the closest narrower consistent scale;
        if no scale reaches the noise level the assumed PSF is KEPT (chi2
        alone cannot separate PSF error from model/noise-estimate error).
        """
        ok = lambda r: r["chi2"] <= self.tau * (1 + self.consist_tol)  # noqa: E731
        if ok(self._run(1.0, 0.0)):
            return 1.0
        for s in sorted([x for x in self.scales if x < 1.0], reverse=True):
            if ok(self._run(s, 0.0)):
                return s
        return 1.0

    def select_lambda(self, s):
        """
        Stage 2: largest lambda whose run reaches the discrepancy.
        If none does (data inconsistent with the model, e.g. wrong PSF), take
        the largest lambda whose chi2 is within fit_tol of the best chi2.
        """
        for lam in sorted(self.lambdas, reverse=True):
            r = self._run(s, lam, snaps=(self.fixed_iters,))
            if r["reached"]:
                return lam
        chi2 = {lam: self._run(s, lam)["chi2"] for lam in [0.0] + self.lambdas}
        limit = (1 + self.fit_tol) * min(chi2.values())
        return max(lam for lam, c in chi2.items() if c <= limit)

    @staticmethod
    def _pack(r, s, lam, noise, extra=None):
        info = {"scale": s, "lambda": lam, "iterations": r["iterations"],
                "reached": bool(r["reached"]), "stop_reason": r["stop_reason"], "chi2": r["chi2"],
                "whiteness": r["whiteness"], "noise_a": noise["a"],
                "noise_b": noise["b"], "noise_sigma": noise["sigma_global"]}
        if extra:
            info.update(extra)
        return info

    def _final(self, s, lam, fixed=False):
        if fixed:
            r = run_rl_tv(
                self.g,
                self._otf(s),
                self.noise,
                lam,
                n_max=self.fixed_iters,
                tau=-1.0,
                plateau_tol=0.0,
            )

            info = self._pack(
                r, s, lam, self.noise,
                {
                    "iterations": self.fixed_iters,
                    "stopping": "fixed",
                },
            )

            return r["volume"], info

        r = self._run(s, lam)

        return (
            r["volume"],
            self._pack(
                r, s, lam, self.noise,
                {"stopping": r["stop_reason"]},
            ),
        )

    # ---- public API ------------------------------------------------
    def restore(self, g):
        vol, info, _ = self._restore_with_variants(g, ablations=False)
        return vol, info

    def variants(self, g):
        _, _, out = self._restore_with_variants(g, ablations=True)
        return out

    def _restore_with_variants(self, g, ablations):
        self._reset(np.asarray(g, dtype=np.float32))
        self.log(f"   noise model: a={self.noise['a']:.3e} b={self.noise['b']:.3e} "
                 f"sigma={self.noise['sigma_global']:.4f}")
        s_star = self.select_scale()
        lam_star = self.select_lambda(s_star)
        full = self._final(s_star, lam_star)
        full[1]["runs"] = list(self.runs)
        self.log(f"   => selected PSF scale={s_star:g}, lambda={lam_star:g}, "
                 f"iterations={full[1]['iterations']}")
        out = {"adaptive_full": full}
        if not ablations:
            return full[0], full[1], out

        lam_s1 = self.select_lambda(1.0)
        out["no_psf_awareness"] = self._final(1.0, lam_s1)
        out["no_lambda_awareness"] = self._final(s_star, self.default_lambda)
        out["no_adaptive_stop"] = self._final(
            s_star, lam_star, fixed=True
        )
        out["stop_only_RL"] = self._final(1.0, 0.0)
        for v in out.values():
            v[1]["runs"] = list(self.runs)
        return full[0], full[1], out


# =====================================================
# SELF-TEST (synthetic, reference used ONLY to check behaviour)
# =====================================================

def synthetic_bars(shape=(24, 64, 64)):
    v = np.zeros(shape, np.float32)
    z0, z1 = shape[0] // 4, 3 * shape[0] // 4
    for y0 in (10, 24, 38):
        v[z0:z1, y0:y0 + 6, 8:shape[2] - 8] = 1.0
    return v


def gaussian_psf(shape, sig=(2.0, 1.4, 1.4)):
    zz, yy, xx = np.meshgrid(*[np.arange(n) - n // 2 for n in shape],
                             indexing="ij")
    p = np.exp(-(zz**2 / (2 * sig[0]**2) + yy**2 / (2 * sig[1]**2)
                 + xx**2 / (2 * sig[2]**2)))
    return normalize_psf(p)


def selftest():
    rng = np.random.default_rng(0)
    shape = (24, 64, 64)
    truth = synthetic_bars(shape)
    psf = gaussian_psf(shape)
    H = make_otf(psf, shape)
    blur = np.maximum(conv(truth, H), 0)

    # --- 1. noise model recovers a, b ---
    peak, read = 40.0, 0.01
    a_true = 1.0 / peak
    obs = rng.poisson(peak * blur) / peak + rng.normal(0, read, shape)
    obs = (obs + 0.05).astype(np.float32)          # offset keeps >0 (no clipping)
    nm = estimate_noise_model(obs)
    print(f"noise est a={nm['a']:.4f} (true {a_true:.4f}), "
          f"b={nm['b']:.2e} (true {read**2:.2e})")
    assert 0.4 * a_true < nm["a"] < 2.5 * a_true
    print("PASS Poisson coefficient coarse check; read-noise intercept not validated")

    # --- 2. scale_psf behaves ---
    assert np.allclose(scale_psf(psf, 1.0), psf)
    def width(p):
        z = np.arange(shape[2]) - shape[2] // 2
        m = p.sum(axis=(0, 1))
        return float(np.sqrt(np.sum(m * z**2) / m.sum()))
    w = [width(scale_psf(psf, s)) for s in (0.8, 1.0, 1.25)]
    assert w[0] < w[1] < w[2], w
    assert abs(scale_psf(psf, 1.25).sum() - 1) < 1e-5
    print("PASS scale_psf widths", [round(x, 2) for x in w])

    # --- 3. discrepancy stopping works and output is valid ---
    obs = np.maximum(obs - 0.05, 0).astype(np.float32)
    r = run_rl_tv(obs, H, estimate_noise_model(obs), 0.0, n_max=60)
    assert np.isfinite(r["volume"]).all() and r["volume"].min() >= 0
    print(f"PASS RL stop: it={r['iterations']} reached={r['reached']} "
          f"chi2={r['chi2']:.3f}")

    # --- 4. lambda=0 matches plain RL loop ---
    f = np.full_like(obs, np.maximum(obs, EPS).mean())
    gg = np.maximum(obs, EPS)
    for _ in range(8):
        f = np.maximum(f * corr(gg / np.maximum(conv(f, H), EPS), H), 0)
    r8 = run_rl_tv(obs, H, estimate_noise_model(obs), 0.0, n_max=8,
                   tau=-1.0, snap_iters=(8,))
    err = np.max(np.abs(r8["snaps"][8] - f)) / (f.max() + 1e-12)
    assert err < 1e-4, err
    print(f"PASS lambda=0 equals plain RL (rel err {err:.1e})")

    # --- 5. full controller, matched and mismatched PSF ---
    from skimage.metrics import structural_similarity as ssim
    for mism in (1.0, 1.25):
        assumed = scale_psf(psf, mism)
        dec = AdaptiveDeconvolver(assumed, n_max=60, log=lambda *_: None)
        vol, info = dec.restore(obs)
        s_in = ssim(truth, np.clip(obs, 0, 1), data_range=1.0)
        s_out = ssim(truth, np.clip(vol, 0, 1), data_range=1.0)
        print(f"PASS controller mismatch={mism}: chose scale={info['scale']:g} "
              f"lam={info['lambda']:g} it={info['iterations']} | "
              f"SSIM input {s_in:.3f} -> {s_out:.3f}")
        assert np.isfinite(vol).all() and vol.min() >= 0
        if mism == 1.0:
            assert info["scale"] == 1.0, info["scale"]
        else:
            assert info["scale"] < 1.0, info["scale"]

    # Regression: fixed stopping must work for lambda 0 and >0.
    for lam in (0.0, 0.01):
        dec = AdaptiveDeconvolver(
            psf, n_max=60, fixed_iters=12,
            log=lambda *_: None,
        )
        dec._reset(obs)

        x, info = dec._final(1.0, lam, fixed=True)

        direct = run_rl_tv(
            obs, dec._otf(1.0), dec.noise, lam,
            n_max=12, tau=-1.0, plateau_tol=0.0,
        )

        assert info["iterations"] == 12
        assert info["stopping"] == "fixed"

        np.testing.assert_allclose(
            x, direct["volume"],
            atol=1e-6, rtol=1e-6,
        )

    print("PASS genuine fixed-iteration ablation")


if __name__ == "__main__":
    selftest()