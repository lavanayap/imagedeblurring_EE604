"""
Verify RL-TV outputs against the team's Wiener results and expected formats.

Run from project root AFTER rl_tv_final.py has finished:
    python rl_tv/verify_rl_tv.py
Optional:
    python rl_tv/verify_rl_tv.py --rl-dir rl_tv/results_final --wiener-dir wiener/results_final
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wiener"))
from wiener_final import (  # noqa: E402
    DATA_FILES, load_zip_volume, center_crop, create_regions, calculate_metrics,
)

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def warn(msg):
    print(f"[WARN] {msg}")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(a):
    rl_dir, w_dir, data_dir = Path(a.rl_dir), Path(a.wiener_dir), Path(a.data_dir)

    # ---------- 1. files exist ----------
    required = ["rl_tv_summary.csv", "experiment_manifest.json"]
    for c in ("low", "high"):
        required += [f"{c}/best_rl_tv.tif", f"{c}/best_rl.tif",
                     f"{c}/validation_sweep.csv",
                     f"{c}/relative_change_per_iteration.csv"]
    for r in required:
        check(f"exists: {r}", (rl_dir / r).exists())
    if not all(results):
        print("\nMissing files; fix those first.")
        return 1

    manifest = json.loads((rl_dir / "experiment_manifest.json").read_text())
    summary = read_csv(rl_dir / "rl_tv_summary.csv")
    shape = tuple(manifest["volume_shape_zyx"])

    # ---------- 2. manifest vs Wiener manifest ----------
    wm_path = w_dir / "experiment_manifest.json"
    if wm_path.exists():
        wm = json.loads(wm_path.read_text())
        check("volume shape matches Wiener",
              manifest["volume_shape_zyx"] == wm["volume_shape_zyx"],
              f"{manifest['volume_shape_zyx']} vs {wm['volume_shape_zyx']}")
        check("regions_y match Wiener",
              manifest["regions_y"] == wm["regions_y"],
              f"{manifest['regions_y']}")
        check("FFT boundary matches", manifest["fft_boundary"] == wm["fft_boundary"])
        check("metric range/clipping/SSIM dims match",
              manifest["evaluation"]["metric_data_range"] == wm["evaluation"]["metric_data_range"]
              and manifest["evaluation"]["metric_clipping"] == wm["evaluation"]["metric_clipping"]
              and manifest["evaluation"]["ssim_dimensions"] == wm["evaluation"]["ssim_dimensions"])
    else:
        warn(f"{wm_path} not found; skipping manifest comparison")

    # ---------- 3. calibration + input metrics vs Wiener ----------
    ws_path = w_dir / "wiener_summary.csv"
    wsum = {r["condition"]: r for r in read_csv(ws_path)} if ws_path.exists() else {}
    if not wsum:
        warn(f"{ws_path} not found; skipping input-metric comparison")

    for c in ("low", "high"):
        # calibration slope/intercept
        wmeta = w_dir / c / "metadata.json"
        if wmeta.exists():
            wc = json.loads(wmeta.read_text())["calibration"]
            mc = manifest["conditions"][c]["calibration"]
            check(f"{c}: calibration slope/intercept match Wiener",
                  np.isclose(wc["slope"], mc["slope"], rtol=1e-6)
                  and np.isclose(wc["intercept"], mc["intercept"], atol=1e-8),
                  f"slope {mc['slope']:.6f} vs {wc['slope']:.6f}")
        # input metrics
        if c in wsum:
            for row in (r for r in summary if r["condition"] == c):
                ok = (np.isclose(float(row["input_PSNR"]), float(wsum[c]["input_PSNR"]), atol=1e-6)
                      and np.isclose(float(row["input_SSIM"]), float(wsum[c]["input_SSIM"]), atol=1e-6))
                check(f"{c}/{row['method']}: input PSNR/SSIM equal Wiener's",
                      ok, f"PSNR {float(row['input_PSNR']):.5f} vs "
                          f"{float(wsum[c]['input_PSNR']):.5f}; "
                          f"SSIM {float(row['input_SSIM']):.5f} vs "
                          f"{float(wsum[c]['input_SSIM']):.5f}")

    # ---------- 4. output file format vs best_wiener.tif ----------
    for c in ("low", "high"):
        for name in ("best_rl_tv.tif", "best_rl.tif"):
            v = tifffile.imread(rl_dir / c / name)
            check(f"{c}/{name}: shape {v.shape} == {shape}", v.shape == shape)
            check(f"{c}/{name}: float32", v.dtype == np.float32, str(v.dtype))
            check(f"{c}/{name}: finite and >= 0",
                  bool(np.isfinite(v).all() and v.min() >= 0),
                  f"min={v.min():.4g} max={v.max():.4g}")
            if v.max() > 5:
                warn(f"{c}/{name}: max {v.max():.3g} is far above 1 (expected ~[0,1])")
        wt = w_dir / c / "best_wiener.tif"
        if wt.exists():
            w = tifffile.imread(wt)
            check(f"{c}: same shape and dtype as best_wiener.tif",
                  w.shape == v.shape and w.dtype == v.dtype,
                  f"{w.shape}/{w.dtype} vs {v.shape}/{v.dtype}")

    # ---------- 5. recompute test metrics from SAVED tifs ----------
    ref_full = load_zip_volume(data_dir / DATA_FILES["reference"])
    reference = center_crop(ref_full / 65535.0, shape)
    regions = create_regions(reference)
    for row in summary:
        key = "best_rl_tv.tif" if row["method"] == "RL-TV" else "best_rl.tif"
        v = tifffile.imread(rl_dir / row["condition"] / key)
        m = calculate_metrics(reference, v, regions["test"])
        check(f"{row['condition']}/{row['method']}: CSV test metrics == recomputed from saved tif",
              np.isclose(m["psnr_db"], float(row["test_PSNR"]), atol=1e-4)
              and np.isclose(m["ssim"], float(row["test_SSIM"]), atol=1e-4),
              f"PSNR {m['psnr_db']:.4f}/{float(row['test_PSNR']):.4f} "
              f"SSIM {m['ssim']:.4f}/{float(row['test_SSIM']):.4f}")

    # ---------- 6. selection sanity ----------
    lams = manifest["lambdas"]
    its = manifest["iteration_checkpoints"]
    for row in summary:
        lam, it = float(row["lambda"]), int(row["iterations"])
        if row["method"] == "RL-TV" and lam in (min(x for x in lams if x > 0), max(lams)):
            warn(f"{row['condition']}: best lambda={lam:g} is at the edge of the grid; extend it")
        if it == max(its):
            warn(f"{row['condition']}/{row['method']}: best iterations={it} is the max tested; try more")
        if float(row["PSNR_gain"]) < 0:
            warn(f"{row['condition']}/{row['method']}: PSNR is BELOW input (gain {float(row['PSNR_gain']):.3f} dB), "
                 "check profile plot for over-smoothing/artefacts")

    n_fail = results.count(False)
    print(f"\n{results.count(True)} passed, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rl-dir", default="rl_tv/results_final")
    p.add_argument("--wiener-dir", default="wiener/results_final")
    p.add_argument("--data-dir", default="data/raw")
    sys.exit(main(p.parse_args()))