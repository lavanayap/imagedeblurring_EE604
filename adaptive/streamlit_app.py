"""Run: streamlit run streamlit_app.py -- --model-dir saved_model"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tifffile
from adaptive_core import (AdaptiveDeconvolver, estimate_noise_model,
                           make_otf, normalize_psf, run_rl_tv, scale_psf)


def load_volume(upload):
    raw = tifffile.imread(io.BytesIO(upload.getvalue()))
    if raw.ndim != 3:
        raise ValueError(f'Expected 3D TIFF (Z,Y,X), got {raw.shape}')
    if np.issubdtype(raw.dtype, np.integer):
        vol = raw.astype(np.float32) / float(np.iinfo(raw.dtype).max)
    else:
        vol = raw.astype(np.float32)
    if not np.isfinite(vol).all() or vol.min() < 0 or vol.max() > 1:
        raise ValueError('Input must contain finite intensities in [0,1] after scaling')
    return vol


@st.cache_resource
def load_model(model_dir):
    d = Path(model_dir)
    config = json.loads((d / 'model_config.json').read_text())
    psf = normalize_psf(np.load(d / config['psf_file']))
    return config, psf


def restore_fixed(vol, config, psf):
    # Disable adaptive stopping to honor the dataset-level iteration preset.
    H = make_otf(scale_psf(psf, config['psf_scale']), vol.shape)
    noise = estimate_noise_model(vol)  # diagnostics only when tau=-1
    result = run_rl_tv(vol, H, noise, lam=config['lambda_tv'],
                       n_max=config['iterations'], tau=-1.0,
                       plateau_tol=0.0)
    return result['volume'], {'scale': config['psf_scale'],
                              'lambda': config['lambda_tv'],
                              'iterations': result['iterations']}


def restore_adaptive(vol, config, psf):
    # Optional: keeps per-image adaptation from the original algorithm.
    model = AdaptiveDeconvolver(psf,
        n_max=config['calibration']['n_max'],
        tau=config['calibration']['tau'],
        plateau_tol=config['calibration']['plateau_tol'],
        log=lambda *_: None)
    return model.restore(vol)


def main():
    parser = argparse.ArgumentParser()
    BASE_DIR = Path(__file__).resolve().parent

    parser.add_argument(
        "--model-dir",
        default=str(BASE_DIR / "saved_model")
    )    
    args, _ = parser.parse_known_args()
    st.title('3D Adaptive RL-TV Microscopy Deconvolution')
    try:
        config, psf = load_model(args.model_dir)
    except Exception as exc:
        st.error(f'Cannot load saved configuration: {exc}')
        st.stop()
    st.caption(f"Dataset preset calibrated on {config['calibration']['num_images']} images")
    mode = st.radio('Restoration mode', ['Dataset preset (fast, fixed)', 'Per-image adaptive (slower)'])
    upload = st.file_uploader('Upload a 3D TIFF volume', type=['tif', 'tiff'])
    if upload is None:
        st.stop()
    try:
        vol = load_volume(upload)
        expected_shape = tuple(config['input']['shape_zyx'])
        if vol.shape != expected_shape:
            raise ValueError(f'Expected (Z,Y,X) {expected_shape}, got {vol.shape}. Prepare the same crop and PSF shape.')
        if st.button('Restore', type='primary'):
            with st.spinner('Restoring image...'):
                if mode.startswith('Dataset'):
                    restored, selected = restore_fixed(vol, config, psf)
                else:
                    restored, selected = restore_adaptive(vol, config, psf)
            st.json({k:v for k,v in selected.items() if k in ('scale','lambda','iterations')})
            z = vol.shape[0] // 2
            col1, col2 = st.columns(2)
            col1.image(vol[z], caption='Input middle Z slice', clamp=True)
            col2.image(restored[z], caption='Restored middle Z slice', clamp=True)
            buf = io.BytesIO()
            tifffile.imwrite(buf, restored.astype(np.float32), photometric='minisblack')
            st.download_button('Download restored 3D TIFF', buf.getvalue(),
                               file_name='restored_3d.tif', mime='image/tiff')
    except Exception as exc:
        st.error(str(exc))

if __name__ == '__main__':
    main()
