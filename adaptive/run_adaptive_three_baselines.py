"""
ADAPTIVE STRATEGY EXPERIMENT (3D)  --  runs after wiener_final.py / rl_tv_final.py
===================================================================================
Project: Noise-Adaptive and PSF-Robust Fluorescence Microscopy Deconvolution
Dataset: EPFL Synthetic Hollow Bars

Uses the SAME protocol as the teammates' scripts (imported, not copied):
loader, centre crop, reference-assisted affine calibration, Y-regions and
PSNR / 3D-SSIM metrics (functions from wiener/wiener_final.py).

Compares, on identical data and the identical test region:
  input                       calibrated degraded volume
  inverse (truncated)         [oracle-tuned on validation]   (if inverse/ importable)
  wiener                      [oracle-tuned K on validation]
  rl_oracle / rltv_oracle     [iterations / lambda tuned on validation with the
                               reference = best case for a FIXED strategy]
  rl_default / rltv_default   [fixed, untuned settings = realistic fixed use]
  adaptive_full               [PROPOSED: no reference used, nothing tuned]
  ablations                   no_psf_awareness, no_lambda_awareness,
                              no_adaptive_stop, stop_only_RL

PSF-mismatch experiment (--mismatch 0.8 1.0 1.2): the PSF given to EVERY method
is the true PSF rescaled by that factor, while the data stay the same.

Run from the project root:
    python adaptive/adaptive_core.py --selftest        (or just run the file)
    python adaptive/run_adaptive.py --fast             # quick check
    python adaptive/run_adaptive.py                    # full experiment
    python adaptive/run_adaptive.py --mismatch 0.8 1.0 1.2   # PSF robustness

Limitations (state in report): one specimen, spatial holdouts; the calibrated
input uses reference-assisted affine calibration (shared with teammates) while
the adaptive CONTROLLER itself is reference-free; periodic boundaries; Poisson
RL on clipped, read-noise-contaminated data is approximate.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
for sub in ("wiener", "inverse", "adaptive"):
    sys.path.insert(0, str(ROOT / sub))

from wiener_final import (  # noqa: E402
    DATA_FILES, load_zip_volume, center_crop, normalize_psf,
    psf_peak_offset, create_regions, calibrate_observation,
    calculate_metrics, save_csv, save_json, Wiener3D,
)
from adaptive_core import (  # noqa: E402
    AdaptiveDeconvolver, run_rl_tv, make_otf, scale_psf,
    estimate_noise_model,
)

try:                                           # teammate's inverse filter
    from inverse_deconvolution import InverseFilter
except Exception as exc:                       # noqa: BLE001
    InverseFilter = None
    print(f"[info] inverse baseline skipped ({exc})")

WIENER_K = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
INVERSE_T = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
CHECKPOINTS = [5, 10, 20, 30, 50, 100, 150, 200]
ORACLE_LAMBDAS = [1e-3, 1e-2, 3e-2]
DEFAULT_LAMBDA = 1e-2


# =====================================================
# FIXED BASELINES
# =====================================================

def fixed_rl_family(g, H, reference, regions, lambdas, checkpoints,
                    n_max, noise):
    """
    Run fixed RL-TV with NO stopping rule; evaluate at checkpoints.
    Returns best-by-validation (oracle) and fixed-default volumes.
    """
    best = {"rltv": None}
    default = {"rltv": None}
    timing = {}
    for lam in list(lambdas):
        t0 = time.perf_counter()
        r = run_rl_tv(g, H, noise, lam, n_max=n_max, tau=-1.0,
                      snap_iters=tuple(checkpoints))
        dt = time.perf_counter() - t0
        key = "rl" if lam == 0 else "rltv"
        for it, vol in r["snaps"].items():
            m = calculate_metrics(reference, vol, regions["validation"])
            score = (m["ssim"], m["psnr_db"])
            if best[key] is None or score > best[key]["score"]:
                best[key] = {"score": score, "lam": lam, "iters": it,
                             "volume": vol.copy(),
                             "runtime": dt * it / n_max}
            if (lam == 0 and it == 30) or \
                    (lam == DEFAULT_LAMBDA and it == 50):
                default[key] = {"lam": lam, "iters": it, "volume": vol.copy(),
                                "runtime": dt * it / n_max}
        print(f"   fixed RL family lam={lam:g} done ({dt:.1f}s)", flush=True)
        del r
    return best, default


def best_wiener(g, psf, reference, regions):
    eng = Wiener3D(g, psf)
    best = None
    for K in WIENER_K:
        t0 = time.perf_counter()
        vol = eng.restore(K)
        dt = time.perf_counter() - t0
        m = calculate_metrics(reference, vol, regions["validation"])
        if best is None or (m["ssim"], m["psnr_db"]) > best["score"]:
            best = {"score": (m["ssim"], m["psnr_db"]), "K": K,
                    "volume": vol, "runtime": dt}
    return best


def best_inverse(g, psf, reference, regions):
    if InverseFilter is None:
        return None
    filt = InverseFilter(psf, g.shape)
    best = None
    for t in INVERSE_T:
        t0 = time.perf_counter()
        vol, _ = filt.restore(g, threshold=t, mode="truncated")
        dt = time.perf_counter() - t0
        m = calculate_metrics(reference, vol, regions["validation"])
        if best is None or (m["ssim"], m["psnr_db"]) > best["score"]:
            best = {"score": (m["ssim"], m["psnr_db"]), "threshold": t,
                    "volume": vol, "runtime": dt}
    return best


# =====================================================
# PLOTS
# =====================================================

def plot_profile(reference, volumes, path, title):
    z, x = reference.shape[0] // 2, reference.shape[2] // 2
    pos = np.arange(reference.shape[1])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(pos, reference[z, :, x], "k-", lw=2, label="Reference")
    for name, vol in volumes.items():
        ax.plot(pos, vol[z, :, x], label=name, lw=1.2)
    ax.set_xlabel("Y (pixels)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_slices(reference, volumes, path, title):
    names = ["Reference"] + list(volumes)
    vols = [reference] + list(volumes.values())
    fig, axes = plt.subplots(1, len(vols), figsize=(3.2 * len(vols), 3.6))
    for ax, n, v in zip(np.atleast_1d(axes), names, vols):
        ax.imshow(v[v.shape[0] // 2], cmap="gray", vmin=0, vmax=1)
        ax.set_title(n, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_controller(dec, selected, path, title):
    """Left: PSF-scale whiteness. Right: chi2 trace of the selected run."""
    scales, chi2 = [], []
    for (s, lam), run in dec.cache.items():
        if lam == 0.0:
            scales.append(s)
            chi2.append(run["chi2"])
    order = np.argsort(scales)
    scales = np.array(scales)[order]
    chi2 = np.array(chi2)[order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(scales, chi2, "o-")
    axes[0].axhline(dec.tau, ls="--", color="red", label=f"tau={dec.tau:g}")
    axes[0].axvline(selected["scale"], ls=":", color="green",
                    label=f"selected s={selected['scale']:g}")
    axes[0].set_xlabel("PSF width scale s (relative to assumed PSF)")
    axes[0].set_ylabel("chi2 at stop / n_max (RL, lambda=0)")
    axes[0].set_title("PSF-consistency stage")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    key = (selected["scale"], selected["lambda"])
    tr = dec.cache[key]["trace"]
    axes[1].plot([t["iteration"] for t in tr], [t["chi2"] for t in tr])
    axes[1].axhline(dec.tau, ls="--", color="red", label=f"tau={dec.tau:g}")
    axes[1].axvline(selected["iterations"], ls=":", color="green",
                    label=f"stop at it={selected['iterations']}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("RL-TV iteration")
    axes[1].set_ylabel("chi2 (signal voxels)")
    axes[1].set_title(f"Discrepancy stopping (lambda={selected['lambda']:g})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mismatch(summary_rows, out_dir):
    mism = sorted({r["psf_mismatch"] for r in summary_rows})
    if len(mism) < 2:
        return
    methods = ["input", "wiener_oracle", "rltv_oracle",
               "rltv_default", "adaptive_full"]
    for cond in ("low", "high"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in methods:
            ys = []
            for mm in mism:
                v = [r["test_SSIM"] for r in summary_rows
                     if r["condition"] == cond and r["method"] == m
                     and r["psf_mismatch"] == mm]
                ys.append(v[0] if v else np.nan)
            ax.plot(mism, ys, "o-", label=m,
                    lw=2.5 if m == "adaptive_full" else 1.2)
        ax.set_xlabel("PSF mismatch factor given to the methods")
        ax.set_ylabel("Test SSIM")
        ax.set_title(f"{cond.upper()} SNR: robustness to PSF error")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"psf_mismatch_{cond}.png", dpi=160)
        plt.close(fig)


# =====================================================
# MAIN
# =====================================================

def run(args):
    data_dir, out_dir = Path(args.data_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = (args.z, args.y, args.x)

    print("Loading reference and PSF...", flush=True)
    reference = center_crop(
        load_zip_volume(data_dir / DATA_FILES["reference"]) / 65535.0, shape)
    psf_true = normalize_psf(center_crop(
        normalize_psf(load_zip_volume(data_dir / DATA_FILES["psf"])), shape))
    print("Volume shape:", shape, "| PSF peak offset:",
          psf_peak_offset(psf_true), flush=True)
    regions = create_regions(reference)

    summary, manifest = [], {}
    observations = {}
    for cond in ("low", "high"):
        obs = center_crop(
            load_zip_volume(data_dir / DATA_FILES[cond]) / 65535.0, shape)
        cal, slope, icpt = calibrate_observation(
            obs, reference, regions["calibration"])
        observations[cond] = (cal, slope, icpt)
        del obs

    for mism in args.mismatch:
        psf = scale_psf(psf_true, mism)          # PSF handed to ALL methods
        H = make_otf(psf, shape)
        for cond in ("low", "high"):
            tag = f"mismatch_{mism:g}/{cond}"
            cdir = out_dir / f"mismatch_{mism:g}" / cond
            cdir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'=' * 60}\n{tag.upper()}\n{'=' * 60}", flush=True)

            cal, slope, icpt = observations[cond]
            volumes_for_plots = {}

            def record(method, family, vol, params, iterations, runtime):
                val = calculate_metrics(reference, vol, regions["validation"])
                tst = calculate_metrics(reference, vol, regions["test"])
                summary.append({
                    "psf_mismatch": mism, "condition": cond, "method": method,
                    "family": family, "params": params,
                    "iterations": iterations,
                    "runtime_s": round(runtime, 3),
                    "val_PSNR": val["psnr_db"], "val_SSIM": val["ssim"],
                    "test_PSNR": tst["psnr_db"], "test_SSIM": tst["ssim"],
                })
                print(f"{method:>22s} | test PSNR={tst['psnr_db']:.3f} "
                      f"SSIM={tst['ssim']:.5f} | val SSIM={val['ssim']:.5f} "
                      f"| {params}", flush=True)

            record("input", "none", cal, "calibrated input", 0, 0.0)

            # ---------- fixed baselines ----------
            if not args.skip_baselines:
                w = best_wiener(cal, psf, reference, regions)
                record("wiener_oracle", "oracle_tuned", w["volume"],
                       f"K={w['K']:g}", 0, w["runtime"])
                volumes_for_plots["Wiener"] = w["volume"]
                inv = best_inverse(cal, psf, reference, regions)
                if inv is not None:
                    record("inverse_oracle", "oracle_tuned", inv["volume"],
                           f"thr={inv['threshold']:g}", 0, inv["runtime"])
                noise = estimate_noise_model(cal)
                best, default = fixed_rl_family(
                    cal, H, reference, regions, ORACLE_LAMBDAS, CHECKPOINTS,
                    max(CHECKPOINTS), noise)
                b = best["rltv"]
                record(
                    "rltv_oracle", "oracle_tuned",
                    b["volume"],
                    f"lam={b['lam']:g}",
                    b["iters"], b["runtime"]
                )
                volumes_for_plots["RL-TV (tuned)"] = b["volume"]

                d = default["rltv"]
                record(
                    "rltv_default", "fixed_default",
                    d["volume"],
                    f"lam={d['lam']:g}",
                    d["iters"], d["runtime"]
                )
                volumes_for_plots["RL-TV (fixed)"] = d["volume"]
                del best, default

            # ---------- PROPOSED adaptive method ----------
            print("-- adaptive controller (no reference used) --", flush=True)
            dec = AdaptiveDeconvolver(
                psf, scales=tuple(args.scales), lambdas=tuple(args.lambdas),
                n_max=args.n_max, tau=args.tau,
                default_lambda=DEFAULT_LAMBDA, fixed_iters=50,
                plateau_tol=args.plateau_tol,
                plateau_min_iter=args.plateau_min_iter)
            t0 = time.perf_counter()
            if args.no_ablation:
                vol, info = dec.restore(cal)
                variants = {"adaptive_full": (vol, info)}
            else:
                variants = dec.variants(cal)
            total_t = time.perf_counter() - t0

            for name, (vol, info) in variants.items():
                if name == "stop_only_RL":
                    continue
                fam = "proposed" if name == "adaptive_full" else "ablation"
                record(name, fam, vol,
                       f"s={info['scale']:g} lam={info['lambda']:g} "
                       f"{info['stopping']}", info["iterations"], total_t)
            full_vol, full_info = variants["adaptive_full"]
            volumes_for_plots["Adaptive"] = full_vol

            # ---------- outputs ----------
            tifffile.imwrite(cdir / "adaptive_full.tif", full_vol)
            save_csv(cdir / "controller_runs.csv", full_info["runs"])
            plot_controller(dec, full_info, cdir / "controller_diagnostics.png",
                            f"{tag}: adaptive controller diagnostics")
            plot_profile(reference, {"Input": cal, **volumes_for_plots},
                         cdir / "bar_profile.png", f"{tag}: central bar profile")
            plot_slices(reference, {"Input": cal, **volumes_for_plots},
                        cdir / "slices.png", f"{tag}: central XY slice")
            manifest[tag] = {
                "calibration": {"slope": slope, "intercept": icpt},
                "noise_model": {k: full_info[k] for k in
                                ("noise_a", "noise_b", "noise_sigma")},
                "selected": {k: full_info[k] for k in
                             ("scale", "lambda", "iterations", "reached",
                              "chi2", "whiteness")},
                "total_adaptive_seconds": total_t,
            }
            del dec, variants, volumes_for_plots, full_vol

    save_csv(out_dir / "adaptive_summary.csv", summary)
    plot_mismatch(summary, out_dir)
    save_json(out_dir / "experiment_manifest.json", {
        "project": "Fluorescence Microscopy Deconvolution",
        "algorithm": "Noise- and PSF-uncertainty-aware adaptive RL-TV",
        "volume_shape_zyx": list(shape),
        "regions_y": {n: [r[1].start, r[1].stop] for n, r in regions.items()},
        "controller": {"scales": args.scales, "lambdas": args.lambdas,
                       "n_max": args.n_max, "tau": args.tau,
                       "plateau_tol": args.plateau_tol,
                       "plateau_min_iter": args.plateau_min_iter,
                       "reference_used_by_controller": False},
        "fixed_baselines": {"wiener_K": WIENER_K, "inverse_thresholds": INVERSE_T,
                            "rl_checkpoints": CHECKPOINTS,
                            "rltv_lambdas": ORACLE_LAMBDAS,
                            
                            "default_rltv": f"lam={DEFAULT_LAMBDA}, 50 it"},
        "evaluation": {"type": "internal_spatial_holdout",
                       "calibration": "reference_assisted (shared)",
                       "metric_range": 1.0, "ssim_dimensions": 3},
        "psf_mismatch_factors": args.mismatch,
        "runs": manifest,
    })

    print("\nSUMMARY (test region)")
    print(f"{'mism':>5} {'cond':>5} {'method':>22} {'PSNR':>8} {'SSIM':>8}")
    for r in summary:
        print(f"{r['psf_mismatch']:5g} {r['condition']:>5} {r['method']:>22} "
              f"{r['test_PSNR']:8.3f} {r['test_SSIM']:8.5f}")
    print("\nADAPTIVE COMPLETE. Results in:", out_dir.resolve(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--output-dir", default="adaptive/results_final")
    # same default crop as wiener_final.py so numbers are comparable
    p.add_argument("--z", type=int, default=64)
    p.add_argument("--y", type=int, default=192)
    p.add_argument("--x", type=int, default=192)
    p.add_argument("--scales", nargs="+", type=float,
                   default=[0.7, 0.8, 0.9, 1.0])
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    p.add_argument("--n-max", type=int, default=200)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--plateau-tol", type=float, default=0.001,
                   help="stop when chi2 improves < this per iteration (0=off)")
    p.add_argument("--plateau-min-iter", type=int, default=50)
    p.add_argument("--mismatch", nargs="+", type=float, default=[1.0])
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--fast", action="store_true",
                   help="small crop, fewer scales/lambdas (smoke test)")
    a = p.parse_args()
    if a.fast:
        a.z, a.y, a.x = 32, 128, 128
        a.scales = [0.8, 0.9, 1.0]
        a.lambdas = [3e-3, 1e-2, 3e-2]
        a.n_max = 60
        a.plateau_min_iter = 30
        CHECKPOINTS[:] = [5, 10, 20, 30, 50, 60]
        ORACLE_LAMBDAS[:] = [1e-3, 1e-2, 1e-1]
    run(a)