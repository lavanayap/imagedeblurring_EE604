"""
Extra analysis of saved RL / RL-TV results (reads files only; changes nothing).

Answers:
  1. Where do the very bright voxels (max ~3.7) come from? Inside the true bars or outside?
  2. Is total intensity (flux) preserved?
  3. What is the bar FWHM (pixels) for reference / input / Wiener / RL / RL-TV?
  4. Profile overlay including Wiener (if wiener results exist).

Run from project root:
    python rl_tv/analyze_rl_tv.py --rl-dir rl_tv/results_final --wiener-dir wiener/results_full
Outputs go to <rl-dir>/analysis/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wiener"))
from wiener_final import (  # noqa: E402
    DATA_FILES, load_zip_volume, center_crop, create_regions,
    calibrate_observation, save_csv,
)


def fwhm_1d(profile):
    """Width (pixels, linear interpolation) of the contiguous region around the
    global maximum where profile >= half of that maximum."""
    p = np.asarray(profile, dtype=np.float64)
    peak = p.max()
    if peak <= 0:
        return np.nan
    half = peak / 2.0
    i = int(np.argmax(p))
    l = i
    while l > 0 and p[l - 1] >= half:
        l -= 1
    r = i
    while r < len(p) - 1 and p[r + 1] >= half:
        r += 1
    left = l - (p[l] - half) / (p[l] - p[l - 1]) if l > 0 and p[l] != p[l - 1] else float(l)
    right = r + (p[r] - half) / (p[r] - p[r + 1]) if r < len(p) - 1 and p[r] != p[r + 1] else float(r)
    return float(right - left)


def median_fwhm(volume, reference, n_z=9, n_x=9):
    """Median FWHM along Y over a grid of (z, x) positions that cross a true bar."""
    zs = np.linspace(volume.shape[0] * 0.3, volume.shape[0] * 0.7, n_z).astype(int)
    xs = np.linspace(volume.shape[2] * 0.3, volume.shape[2] * 0.7, n_x).astype(int)
    vals = []
    for z in zs:
        for x in xs:
            if reference[z, :, x].max() > 0.5:
                vals.append(fwhm_1d(volume[z, :, x]))
    return float(np.nanmedian(vals)) if vals else np.nan, len(vals)


def analyze(condition, rl_dir, w_dir, data_dir, shape, regions, reference, rows, figs):
    obs = center_crop(load_zip_volume(data_dir / DATA_FILES[condition]) / 65535.0, shape)
    calibrated, _, _ = calibrate_observation(obs, reference, regions["calibration"])

    vols = {"Reference": reference, "Input (calibrated)": calibrated}
    for name, f in (("RL", "best_rl.tif"), ("RL-TV", "best_rl_tv.tif")):
        vols[name] = tifffile.imread(rl_dir / condition / f)
    wt = w_dir / condition / "best_wiener.tif"
    if wt.exists():
        vols["Wiener"] = tifffile.imread(wt)
    else:
        print(f"  (no Wiener volume at {wt}; skipped)")

    bar = reference > 0.5
    bar_dilated = bar.copy()
    for ax in range(3):                      # 3-voxel dilation in each axis
        for s in (-3, -2, -1, 1, 2, 3):
            bar_dilated |= np.roll(bar, s, axis=ax)

    print(f"\n===== {condition.upper()} SNR =====")
    print(f"Reference flux (sum) = {reference.sum():.1f}")
    for name, v in vols.items():
        if name == "Reference":
            continue
        flux = float(v.sum())
        hot = v > 1.2
        n_hot = int(hot.sum())
        inside = int((hot & bar_dilated).sum())
        zyx = np.unravel_index(int(np.argmax(v)), v.shape)
        med, n = median_fwhm(v, reference)
        row = {
            "condition": condition, "volume": name,
            "max": float(v.max()), "p99.9": float(np.percentile(v, 99.9)),
            "voxels_gt_1.0": int((v > 1.0).sum()),
            "voxels_gt_1.2": n_hot,
            "gt_1.2_inside_bars(%)": 100.0 * inside / n_hot if n_hot else np.nan,
            "argmax_zyx": str(tuple(int(i) for i in zyx)),
            "flux_ratio_vs_reference": flux / float(reference.sum()),
            "flux_ratio_vs_input": flux / float(calibrated.sum()),
            "median_FWHM_y_px": med, "n_profiles": n,
        }
        rows.append(row)
        print(f"{name:20s} max={row['max']:.2f}  >1.0: {row['voxels_gt_1.0']}  >1.2: {n_hot}"
              f" ({row['gt_1.2_inside_bars(%)']:.0f}% inside/near true bars)  "
              f"flux/ref={row['flux_ratio_vs_reference']:.3f}  FWHM={med:.2f}px")
    med_ref, n = median_fwhm(reference, reference)
    print(f"{'Reference':20s} FWHM={med_ref:.2f}px (n={n} profiles)")
    rows.append({"condition": condition, "volume": "Reference",
                 "max": float(reference.max()), "median_FWHM_y_px": med_ref, "n_profiles": n,
                 "flux_ratio_vs_reference": 1.0})

    # profile overlay
    z, x = reference.shape[0] // 2, reference.shape[2] // 2
    pos = np.arange(reference.shape[1])
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, v in vols.items():
        ax.plot(pos, v[z, :, x], label=name, lw=2 if name == "Reference" else 1.3)
    ax.set_xlabel("Y (pixels)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{condition.upper()} SNR: central bar profile (all methods)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    figs.append((condition, fig))


def main(a):
    rl_dir, w_dir, data_dir = Path(a.rl_dir), Path(a.wiener_dir), Path(a.data_dir)
    manifest = json.loads((rl_dir / "experiment_manifest.json").read_text())
    shape = tuple(manifest["volume_shape_zyx"])
    reference = center_crop(load_zip_volume(data_dir / DATA_FILES["reference"]) / 65535.0, shape)
    regions = create_regions(reference)

    out = rl_dir / "analysis"
    out.mkdir(exist_ok=True)
    rows, figs = [], []
    for c in ("low", "high"):
        analyze(c, rl_dir, w_dir, data_dir, shape, regions, reference, rows, figs)
    for c, fig in figs:
        fig.savefig(out / f"profile_all_methods_{c}.png", dpi=160)
        plt.close(fig)
    save_csv(out / "analysis_summary.csv", rows)
    print("\nSaved to:", out.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rl-dir", default="rl_tv/results_final")
    p.add_argument("--wiener-dir", default="wiener/results_full")
    p.add_argument("--data-dir", default="data/raw")
    main(p.parse_args())