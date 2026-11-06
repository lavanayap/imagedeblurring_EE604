# Noise-Adaptive and PSF-Robust Fluorescence Microscopy Deconvolution

**EE604 – Image Processing Project**

## 1. Project Overview

Fluorescence microscopy is widely used to visualize biological structures at microscopic scales. However, the captured images are often degraded by optical blur, photon noise and sensor noise, resulting in reduced image clarity and structural detail.

Image deconvolution attempts to recover a sharper image from a blurred and noisy observation using knowledge of the microscope's Point Spread Function (PSF).

This project implements and evaluates classical image deconvolution techniques alongside a proposed **Noise-Informed Adaptive 3D Deconvolution Algorithm**.

The primary objective is to investigate whether adaptive parameter selection can improve reconstruction quality while handling variations in image noise and uncertainty in the assumed PSF.

## 2. Image Degradation Model

The observed microscopy image is modeled as:

\[
g = f * h + n
\]

Where:

- **g** – Observed degraded image
- **f** – Original sharp image
- **h** – Point Spread Function (PSF)
- **n** – Observation noise
- **\*** – Convolution operator

The PSF describes how the microscope spreads light originating from a point source. Incorrect PSF estimation can affect the accuracy of image restoration.

The objective of deconvolution is to estimate the original image **f** from the observed image **g** and an assumed PSF.

## 3. Algorithms Implemented

The final primary comparison consists of four algorithms.

### 3.1 Inverse Filtering

Inverse Filtering attempts to recover the original image by reversing the optical blurring operation in the frequency domain.

It is computationally straightforward but can amplify noise when the optical transfer function contains small values.

### 3.2 Wiener Filtering

Wiener Filtering performs frequency-domain restoration by balancing blur inversion against noise amplification.

It provides a regularized alternative to direct inverse filtering and depends on a suitable regularization parameter.

### 3.3 Richardson–Lucy with Total Variation (RL-TV)

RL-TV is an iterative deconvolution method that combines Richardson–Lucy reconstruction with Total Variation regularization.

The RL component reverses image blur iteratively, while TV regularization helps control noise amplification and preserve structural boundaries.

The fixed RL-TV baseline uses predetermined reconstruction parameters.

### 3.4 Proposed Noise-Informed Adaptive Deconvolution

Our proposed algorithm builds on 3D RL-TV reconstruction and introduces automatic parameter control.

Its main components are:

**A. Noise estimation**

The algorithm estimates a Poisson-Gaussian noise model directly from the degraded image:

\[
\operatorname{Var}(I) = aI + b
\]

The estimated noise characteristics are used to guide reconstruction.

**B. Adaptive regularization**

Rather than using a single fixed TV regularization weight, the algorithm evaluates candidate lambda values and selects a parameter based on data-consistency criteria.

**C. Adaptive stopping**

The algorithm monitors the variance-normalized reconstruction residual and stops using either a discrepancy criterion or a convergence-plateau criterion.

This reduces reliance on a fixed iteration count.

**D. PSF uncertainty assessment**

The controller evaluates candidate PSF widths to examine whether the supplied blur model is consistent with the observed data.

This component is designed primarily to detect overly wide assumed PSFs. Its effectiveness in correcting PSF mismatch remains a limitation of the current implementation.

**E. 3D reconstruction**

The implementation operates on volumetric microscopy data, rather than processing each 2D slice independently.

## 4. Dataset

Experiments use the **EPFL Synthetic Hollow Bars** fluorescence microscopy dataset.

The dataset provides a reference volume, degraded observations and a corresponding PSF.

The benchmark uses:

- A reference fluorescence volume
- Two degraded observation conditions, labelled low and high
- A reference PSF
- Three assumed-PSF width factors: 0.8, 1.0 and 1.2

The full benchmark uses a center-cropped volume of **64 × 192 × 192 voxels**.

### PSF Mismatch Experiments

| PSF Factor | Interpretation |
|---|---|
| 0.8 | Assumed PSF is 20% narrower |
| 1.0 | Assumed PSF matches the reference |
| 1.2 | Assumed PSF is 20% wider |

The observed volume remains unchanged across PSF-width experiments. Only the PSF supplied to the restoration algorithms is modified.

## 5. Evaluation Metrics

Two primary quantitative metrics are used.

### Peak Signal-to-Noise Ratio (PSNR)

PSNR measures reconstruction fidelity relative to a reference image.

Higher PSNR indicates lower reconstruction error in terms of mean squared error.

### Structural Similarity Index (SSIM)

SSIM measures structural similarity between the reconstructed image and reference image.

Higher SSIM indicates better preservation of structural information.

The benchmark calculates both metrics on a designated test region of the 3D volume.

## 6. Experimental Results

The following results are averaged across six experimental conditions: two observation conditions and three PSF-width factors.

### Main Comparison

| Algorithm | Average PSNR (dB) | Average SSIM |
|---|---:|---:|
| Inverse Filtering | 14.758 | 0.4814 |
| Wiener Filtering | 17.323 | 0.5971 |
| RL-TV (Fixed) | 19.034 | 0.8439 |
| **Our Adaptive Algorithm** | **19.241** | **0.8708** |

The proposed adaptive algorithm achieves the highest mean PSNR and SSIM among these four evaluated baseline configurations.

Compared with fixed RL-TV, the adaptive method improves average:

- **PSNR by 0.207 dB**
- **SSIM by 0.0269**

These results indicate that adaptive regularization and reconstruction stopping can improve structural reconstruction quality over the fixed RL-TV configuration.

**Comparison methodology:** The Wiener and Inverse Filtering implementations use reference-based validation parameter selection, whereas the RL-TV row uses fixed settings. The adaptive controller does not directly use the reference image for parameter selection. The shared observation preprocessing does, however, include reference-assisted intensity calibration.

The results should therefore be interpreted as a comparison of the specified configurations rather than proof that the adaptive algorithm outperforms every possible tuning of the classical methods.

### Adaptive Performance Under PSF Variations

| Supplied PSF Factor | Low Condition SSIM | High Condition SSIM |
|---|---:|---:|
| 0.8 | 0.85705 | 0.86831 |
| 1.0 | 0.87847 | 0.87172 |
| 1.2 | 0.87803 | 0.87094 |

The adaptive reconstruction maintains relatively strong structural similarity across the three tested PSF assumptions.

However, in the full experiments, the PSF-selection controller retained scale 1.0 in every condition. Thus, the results demonstrate reconstruction behavior under mismatched PSFs, but not successful automatic correction of the mismatch.

### Additional Validation

The original development experiments also included reference-tuned RL and RL-TV comparisons and ablation studies.

Reference-tuned RL-TV achieved higher average SSIM than the proposed method, indicating that additional improvements remain possible.

Ablation results suggest that adaptive stopping contributes more to the observed SSIM improvement than adaptive lambda selection in the tested configuration.

## 7. Repository Structure

The relevant project files are organized as follows:

```text
imagedeblurring_EE604/
│
├── adaptive/
│   ├── adaptive_core.py
│   ├── run_adaptive_three_baselines.py
│   ├── .gitignore
│   └── results_four_methods/
│       ├── adaptive_summary.csv
│       ├── experiment_manifest.json
│       ├── psf_mismatch_low.png
│       ├── psf_mismatch_high.png
│       ├── mismatch_0.8/
│       ├── mismatch_1/
│       └── mismatch_1.2/
│
├── inverse/
│   └── inverse_deconvolution.py
│
├── wiener/
│   └── wiener_final.py
│
├── rl_tv/
│   └── rl_tv_final.py
│
└── data/
    └── raw/
```

### Important Files

| File | Description |
|---|---|
| `adaptive/adaptive_core.py` | Adaptive noise estimation, RL-TV reconstruction, PSF assessment and controller logic |
| `adaptive/run_adaptive_three_baselines.py` | Final benchmark comparing Inverse, Wiener, fixed RL-TV and Adaptive |
| `adaptive/results_four_methods/adaptive_summary.csv` | Quantitative PSNR and SSIM results |
| `adaptive/results_four_methods/experiment_manifest.json` | Benchmark parameters and experimental metadata |
| `adaptive/results_four_methods/psf_mismatch_low.png` | PSF robustness comparison for the low condition |
| `adaptive/results_four_methods/psf_mismatch_high.png` | PSF robustness comparison for the high condition |

**Implementation note:** The final benchmark's fixed RL-TV comparison is computed using the RL-TV reconstruction routine in `adaptive_core.py`. The teammate's separate `rl_tv_final.py` implementation is retained in the repository.

## 8. Installation

Clone the repository:

```bash
git clone https://github.com/lavanayap/imagedeblurring_EE604.git
cd imagedeblurring_EE604
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the Python packages required by the adaptive benchmark:

```bash
pip install numpy scipy matplotlib scikit-image tifffile
```

Other team components may have additional requirements.

### Dataset Setup

Download the EPFL Synthetic Hollow Bars dataset and place the required reference, degraded observation and PSF archives in `data/raw/`.

The exact filenames and paths must match the dataset configuration in `wiener/wiener_final.py`.

The raw microscopy archives and generated full-resolution TIFF volumes are not included in the adaptive results committed to GitHub.

## 9. Running the Adaptive Algorithm

Run all commands from the repository root.

### Test the adaptive implementation

```bash
python adaptive/adaptive_core.py
```

This executes synthetic self-tests for the noise estimator, PSF scaling, reconstruction behavior and fixed-iteration ablation.

### Quick benchmark

```bash
python adaptive/run_adaptive_three_baselines.py --fast --no-ablation --mismatch 1.0 --output-dir adaptive/results_four_methods_fast
```

### Complete benchmark

```bash
python adaptive/run_adaptive_three_baselines.py --no-ablation --mismatch 0.8 1.0 1.2 --output-dir adaptive/results_four_methods
```

The complete benchmark evaluates four algorithms under six experimental conditions.

## 10. Understanding the Generated Results

Each PSF experiment contains separate folders for the two observation conditions:

```text
results_four_methods/
├── mismatch_0.8/
│   ├── low/
│   └── high/
├── mismatch_1/
│   ├── low/
│   └── high/
└── mismatch_1.2/
    ├── low/
    └── high/
```

Each condition generates:

- `bar_profile.png` – Reconstructed intensity-profile comparison
- `slices.png` – Visual comparison of reconstruction outputs
- `controller_diagnostics.png` – PSF-consistency and convergence diagnostics
- `controller_runs.csv` – Tested parameters and controller behavior
- `adaptive_full.tif` – Full adaptive 3D reconstruction, generated locally but excluded from Git

The top-level summary CSV consolidates the quantitative results.

## 11. Limitations

The current implementation has several limitations:

1. **Limited dataset coverage:** Full primary results are obtained from one Hollow Bars specimen under six experimental conditions.
2. **PSF correction:** The current controller did not demonstrate successful automatic PSF-width correction in the full benchmark.
3. **Noise-model accuracy:** The Poisson-Gaussian model is approximate, particularly for clipped or intensity-calibrated observations.
4. **Stopping calibration:** Convergence-plateau stopping frequently determines termination when the discrepancy threshold is not reached.
5. **Reference-assisted preprocessing:** Although the adaptive controller operates without the clean reference, the shared benchmark includes reference-assisted intensity calibration.
6. **Baseline tuning:** Results depend on the selected baseline configurations; reference-tuned RL-TV can achieve higher SSIM.
7. **Generalization:** Further evaluation is needed on additional fluorescence microscopy specimens, noise settings and experimentally measured PSFs.

## 12. Future Scope

Potential improvements include:

- More reliable noise-variance estimation
- Improved PSF mismatch detection and correction
- Additional microscopy datasets and biological structures
- Evaluation of spatially varying PSFs
- Further analysis of resolution recovery and structural separation
- Improved computational efficiency
- Broader comparisons under consistent parameter-selection protocols

## 13. Conclusion

This project demonstrates the application of classical and adaptive deconvolution techniques to 3D fluorescence microscopy.

The proposed noise-informed adaptive RL-TV method achieves an average **PSNR of 19.241 dB** and **SSIM of 0.8708** across six benchmark conditions.

It outperforms the selected fixed RL-TV configuration in both metrics, with particularly meaningful improvement in structural similarity.

The experimental results support the value of adaptive parameter control while identifying PSF correction, noise-model accuracy and evaluation across additional specimens as important areas for further development.

---

**Project:** EE604 – Noise-Adaptive and PSF-Robust Fluorescence Microscopy Deconvolution

**Repository:** https://github.com/lavanayap/imagedeblurring_EE604