# imagedeblurring_EE604
# 3D Richardson–Lucy Deconvolution for Fluorescence Microscopy

## Overview

This project implements a **3D Richardson–Lucy (RL) deconvolution algorithm** to restore blurred and noisy fluorescence microscopy image volumes.

The implementation uses the EPFL Synthetic Hollow Bars dataset to evaluate restoration under two noise conditions:

- Low-noise dataset (G10-P15)
- High-noise dataset (G10-P30)

The project includes PSF normalization, FFT-based 3D convolution, an adjoint operator, reference-assisted calibration, iterative Richardson–Lucy restoration, and quantitative evaluation.

## Dataset

The project uses the **Synthetic Hollow Bars dataset** provided by EPFL.

Official dataset page:  
https://bigwww.epfl.ch/deconvolution/bars/

The dataset contains six parallel hollow bars. The images are blurred using a theoretical microscopy point spread function (PSF) and corrupted with Gaussian and Poisson noise.

### Required files

Download the following four stack ZIP files from the EPFL dataset page:

| File | Description |
|---|---|
| `Bars-stack.zip` | Uncorrupted reference volume |
| `Bars-G10-P15-stack.zip` | Blurred, low-noise volume |
| `Bars-G10-P30-stack.zip` | Blurred, high-noise volume |
| `PSF-Bars-stack.zip` | Point Spread Function |

### Dataset specifications

- Image dimensions: 128 × 256 × 256
- Image format: TIFF
- Bit depth: 16-bit
- Number of channels: 1

### Directory structure

After downloading the files, place them inside `data/raw/`:

```text
imagedeblurring_EE604/
│
├── data/
│   └── raw/
│       ├── Bars-stack.zip
│       ├── Bars-G10-P15-stack.zip
│       ├── Bars-G10-P30-stack.zip
│       └── PSF-Bars-stack.zip
│
├── results/
├── richardson_lucy_3d.py
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** The dataset files are not included in this GitHub repository. Download them separately from EPFL before running the experiments.

## Methodology

The restoration pipeline consists of the following steps:

1. **Data loading:** Load the reference, degraded image volumes, and PSF from the downloaded ZIP files.
2. **Normalization:** Normalize image intensities and normalize the PSF so that its sum equals one.
3. **Forward model:** Implement 3D convolution using FFT.
4. **Adjoint operator:** Implement the corresponding adjoint using the conjugate of the frequency-domain PSF.
5. **Calibration:** Estimate the gain (`alpha`) and background (`beta`) parameters using least squares against the reference volume.
6. **Restoration:** Apply calibrated 3D Richardson–Lucy deconvolution for different iteration counts.
7. **Evaluation:** Compare the restored volumes against the reference using PSNR, SSIM, and MSE.

The current FFT implementation uses circular boundary conditions.

## Installation

Clone the repository:

```bash
git clone https://github.com/lavanayap/imagedeblurring_EE604.git
cd imagedeblurring_EE604
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and place the four dataset ZIP files in `data/raw/` as described above.

## Running the Project

Run the consolidated 3D Richardson–Lucy implementation:

```bash
python richardson_lucy_3d.py
```

The script processes both noise conditions and evaluates restoration at:

- 30 iterations
- 60 iterations
- 100 iterations
- 120 iterations

The best tested output for each noise condition and the corresponding evaluation results are saved in the `results/` directory.

## Experimental Results

The following results were obtained using the reference-assisted calibration procedure.

### Low-noise dataset (G10-P15)

| Metric | Degraded input | 120 iterations |
|---|---:|---:|
| PSNR (dB) | 21.0112 | 27.5514 |
| SSIM | 0.0750 | 0.9631 |
| MSE | 0.007923 | 0.001757 |

### High-noise dataset (G10-P30)

| Metric | Degraded input | 120 iterations |
|---|---:|