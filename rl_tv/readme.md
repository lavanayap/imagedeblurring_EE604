# RL-TV (Richardson-Lucy with Total Variation) deconvolution

Part of the EE604 project *Noise-Adaptive and PSF-Robust Fluorescence Microscopy
Image Deconvolution*. This folder contains the RL-TV branch only. It uses the same
data, calibration, regions and metrics as `wiener/wiener_final.py` (functions are
imported from that file), so numbers are comparable across methods.

## Files

| File | Purpose |
|---|---|
| `rl_tv_final.py` | 3D RL-TV implementation, λ/iteration sweep, evaluation, output writing |
| `verify_rl_tv.py` | 35 automated checks of outputs against the Wiener protocol and formats |
| `analyze_rl_tv.py` | Peak intensity, flux, bar FWHM and profile overlays (optionally with Wiener) |
| `results_final/` | Main run: λ sweep, up to 100 iterations |
| `results_extended/` | Extension: λ = 0 and 1e-4, up to 300 iterations |

## Algorithm

Dey et al. (2006):

    f_{k+1} = f_k / (1 - λ · div(∇f_k / |∇f_k|)) · Hᵀ( g / (H f_k) )

- H is FFT convolution with the normalized PSF, Hᵀ is correlation (conj(H)).
- λ = 0 switches TV off. Those rows are an **ablation** (what TV adds), not the
  team's RL result.
- Initial estimate: constant equal to the mean of the input.
- Safeguards: `eps = 1e-8` on divisions, `eps_tv = 1e-3` in the gradient norm,
  denominator clipped to [0.5, 2].

## Reproduce (run from the repository root)

    python rl_tv/rl_tv_final.py --selftest
    python rl_tv/rl_tv_final.py                                   # results_final
    python rl_tv/rl_tv_final.py --lambdas 0 0.0001 --checkpoints 100 200 300 \
        --output-dir rl_tv/results_extended
    python rl_tv/verify_rl_tv.py --rl-dir rl_tv/results_final --wiener-dir wiener/results_full
    python rl_tv/analyze_rl_tv.py --rl-dir rl_tv/results_final --wiener-dir <wiener results folder>

Requires `data/raw/` with the four EPFL zips. A full run takes roughly 30-40 minutes
on a laptop CPU. The `.tif` volumes are not committed; the commands above regenerate them.

## Protocol

- Dataset: EPFL Synthetic Hollow Bars, volume 128 × 256 × 256 (Z, Y, X).
- Input: the reference-assisted affine-calibrated volume (same as Wiener).
- Regions along Y: calibration 60-102, validation 102-144, test 144-187.
- Parameters (λ, iterations) are selected on the **validation** region by SSIM
  (tie-break PSNR); the test region is evaluated once for the selected setting.
- Metrics: PSNR and 3D SSIM with data range 1 and clipping to [0, 1].

## Results (test region)

Selected on validation: λ = 1e-4 (smallest nonzero λ tested).

| Condition | Setting | PSNR (dB) | SSIM |
|---|---|---|---|
| low SNR | input | 22.606 | 0.5999 |
| low SNR | RL-TV, 100 it | 23.489 | 0.9205 |
| low SNR | RL-TV, 300 it | 23.977 | 0.9346 |
| high SNR | input | 22.740 | 0.6771 |
| high SNR | RL-TV, 100 it | 23.350 | 0.9388 |
| high SNR | RL-TV, 300 it | 23.494 | 0.9536 |

Additional measurements (`analyze_rl_tv.py`, 100 iterations):

| | max intensity (low / high) | bar FWHM px (low / high) |
|---|---|---|
| reference | 1.0 | 5.0 |
| input | 0.22 / 0.19 | 7.4 / 12.1 |
| Wiener (K = 0.01) | 0.43 / 0.52 | 12.3 / 12.4 |
| RL-TV, 100 it | 3.71 / 2.76 | 3.8 / 3.7 |
| RL-TV, 300 it | 7.58 / 5.25 | 2.8 / 3.5 |

## Findings and limitations

- TV regularization gave no measurable benefit in this setup. Validation scores
  worsened for every λ above 1e-4 (tested up to 0.1 at ≤100 iterations), and at the
  selected λ the result is practically identical to λ = 0. TV at larger λ for
  300 iterations was not evaluated in `results_extended`.
- Peak intensities overshoot the reference (max 1.0) by a large factor, growing with
  iteration count. PSNR/SSIM clip to [0, 1] and do not show this.
- The calibrated input already carries about 2.4× the reference's total intensity
  (shared by all methods); its cause has not been established.
- Validation SSIM was still rising at 300 iterations (not converged).
- One specimen, spatial holdout regions only (not independent samples);
  reference-assisted calibration and parameter selection (not blind);
  periodic (FFT) boundaries; standard RL assumes Poisson data while the calibrated
  input contains read noise, so the likelihood is approximate.

## For the adaptive strategy

Per-iteration relative change `‖f_{k+1} − f_k‖/‖f_k‖` is in
`results_*/{low,high}/relative_change_per_iteration.csv`, and metric curves are in
`validation_sweep.csv`. Function signature: `rl_tv(g, psf, lam, n_iter, callback)`
with `callback(k, f, rel_change)` called after every iteration (`rl_tv_iter` takes a
precomputed OTF for repeated runs).