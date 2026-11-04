"""Inspect EPFL Hollow Bars 3D stacks without modifying the original data.

Download the four *-stack.zip files from https://bigwww.epfl.ch/deconvolution/bars/
and place them under data/raw/. Run: python inspect_epfl.py
"""
from pathlib import Path
import io
import json
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'results'
FILES = {
    'Reference': 'Bars-stack.zip',
    'Low SNR': 'Bars-G10-P15-stack.zip',
    'High SNR': 'Bars-G10-P30-stack.zip',
    'PSF': 'PSF-Bars-stack.zip',
}


def load_zip_stack(filename: str) -> np.ndarray:
    path = RAW / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}\nDownload it from https://bigwww.epfl.ch/deconvolution/bars/')
    with zipfile.ZipFile(path) as archive:
        tiff_names = [n for n in archive.namelist() if n.lower().endswith(('.tif', '.tiff')) and not n.startswith('__MACOSX/')]
        if not tiff_names:
            raise ValueError(f'No TIFF files found in {filename}')
        # EPFL offers a single multipage TIFF for each *-stack.zip archive.
        if len(tiff_names) == 1:
            with archive.open(tiff_names[0]) as f:
                vol = tifffile.imread(io.BytesIO(f.read()))
        else:
            raise ValueError(
                f'{filename} contains {len(tiff_names)} TIFF files, expected one multipage TIFF. '
                'Check that you downloaded the Single z-stack version.'
            )
    if vol.ndim != 3:
        raise ValueError(f'Expected a 3D z-stack in {filename}, got shape {vol.shape}')
    if not np.all(np.isfinite(vol)):
        raise ValueError(f'Nonfinite values in {filename}')
    return vol


def save_views(volumes: dict[str, np.ndarray]) -> None:
    reference = volumes['Reference']
    low = volumes['Low SNR']
    high = volumes['High SNR']
    psf = volumes['PSF']
    # Joint reference/input display scale avoids separately normalizing each image.
    display_max = max(1.0, float(np.percentile(np.concatenate([
        a[a > 0][::100] for a in (reference, low, high) if np.any(a > 0)
    ]), 99.5)))
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    for i, (label, vol) in enumerate(volumes.items()):
        mid = vol.shape[0] // 2
        scale = float(np.max(vol)) if label == 'PSF' else display_max
        scale = max(scale, 1.0)
        axes[0, i].imshow(vol[mid], cmap='gray', vmin=0, vmax=scale)
        axes[0, i].set_title(f'{label}: center XY (z={mid})')
        axes[1, i].imshow(np.max(vol, axis=0), cmap='gray', vmin=0, vmax=scale)
        axes[1, i].set_title(f'{label}: XY max projection')
        for r in (0, 1):
            axes[r, i].axis('off')
    fig.savefig(OUT / 'epfl_stack_overview.png', dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)
    volumes = {}
    stats = {}
    for label, filename in FILES.items():
        arr = load_zip_stack(filename)
        volumes[label] = arr
        stats[label] = {
            'file': filename, 'shape_zyx': list(arr.shape), 'dtype': str(arr.dtype),
            'minimum': float(arr.min()), 'maximum': float(arr.max()),
            'mean': float(arr.mean()), 'sum': float(arr.astype(np.float64).sum()),
        }
        print(f'{label:10} shape={arr.shape} dtype={arr.dtype} '
              f'min={arr.min():.2f} max={arr.max():.2f} mean={arr.mean():.2f}')
    if volumes['Reference'].shape != volumes['Low SNR'].shape or volumes['Reference'].shape != volumes['High SNR'].shape:
        raise ValueError('Reference and observations must have the same shape')
    psf_sum = stats['PSF']['sum']
    if psf_sum <= 0:
        raise ValueError('PSF must have positive sum')
    print(f'PSF sum = {psf_sum:.6g}; normalize PSF before convolution')
    (OUT / 'dataset_statistics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    save_views(volumes)
    print(f'\nSaved overview: {OUT / "epfl_stack_overview.png"}')
    print(f'Saved statistics: {OUT / "dataset_statistics.json"}')


if __name__ == '__main__':
    main()
