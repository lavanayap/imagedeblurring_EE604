"""Dataset-level RL-TV preset for the EPFL Hollow Bars ZIP dataset.

Run from project root:
    python adaptive/calibrate_dataset.py

Uses the two degraded noise-condition volumes, not the ground-truth reference.
Reference-free scaling is identical to the companion Streamlit app for uint16 TIFFs.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ADAPTIVE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ADAPTIVE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "wiener"))

from wiener_final import DATA_FILES, load_zip_volume, center_crop  # noqa: E402
from adaptive_core import AdaptiveDeconvolver, normalize_psf  # noqa: E402


def mode(values):
    counts = Counter(values)
    return sorted(counts, key=lambda x: (-counts[x], x))[0]


def main():
    parser = argparse.ArgumentParser(description="Calibrate dataset-level RL-TV preset on EPFL ZIP stacks")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ADAPTIVE_DIR / "saved_model")
    parser.add_argument("--z", type=int, default=64)
    parser.add_argument("--y", type=int, default=192)
    parser.add_argument("--x", type=int, default=192)
    parser.add_argument("--n-max", type=int, default=200)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--plateau-tol", type=float, default=0.001)
    args = parser.parse_args()

    shape = (args.z, args.y, args.x)
    for name in ("psf", "low", "high"):
        path = args.data_dir / DATA_FILES[name]
        if not path.is_file():
            parser.error(f"Missing required dataset archive: {path}")

    print("Loading EPFL PSF ZIP", flush=True)
    psf = normalize_psf(center_crop(normalize_psf(
        load_zip_volume(args.data_dir / DATA_FILES["psf"])), shape))
    rows = []
    for condition in ("low", "high"):
        path = args.data_dir / DATA_FILES[condition]
        print(f"\nProcessing {condition}: {path.name}", flush=True)
        volume = center_crop(load_zip_volume(path) / 65535.0, shape).astype(np.float32)
        if not np.isfinite(volume).all() or volume.min() < 0 or volume.max() > 1:
            raise ValueError(f"Unexpected input intensity range in {path.name}")
        controller = AdaptiveDeconvolver(
            psf, n_max=args.n_max, tau=args.tau,
            plateau_tol=args.plateau_tol,
            log=lambda *values: None)
        _, info = controller.restore(volume)
        row = dict(condition=condition, file=path.name,
                   scale=float(info["scale"]), lambda_tv=float(info["lambda"]),
                   iterations=int(info["iterations"]), chi2=float(info["chi2"]),
                   reached=bool(info["reached"]), stop_reason=str(info["stop_reason"]))
        rows.append(row)
        print(row, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_image_parameters.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "format_version": 1,
        "algorithm": "fixed_rl_tv_dataset_preset",
        "psf_file": "psf.npy",
        "psf_scale": float(mode([r["scale"] for r in rows])),
        "lambda_tv": float(mode([r["lambda_tv"] for r in rows])),
        "iterations": int(round(float(np.median([r["iterations"] for r in rows])))),
        "input": {
            "ndim": 3,
            "shape_zyx": list(shape),
            "normalization": "unsigned_integer_divide_by_dtype_max_or_float_in_0_1",
        },
        "calibration": {
            "dataset": "EPFL Synthetic Hollow Bars",
            "conditions": [r["condition"] for r in rows],
            "num_images": len(rows),
            "n_max": args.n_max,
            "tau": args.tau,
            "plateau_tol": args.plateau_tol,
            "fraction_reached": sum(r["reached"] for r in rows) / len(rows),
            "reference_used": False,
            "note": "Two noise conditions of the same specimen, not independent training images. "
                    "Mode/median values are a heuristic preset, not an optimized model.",
        },
    }
    np.save(args.out / "psf.npy", psf)
    (args.out / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nSaved to: {args.out.resolve()}", flush=True)
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
