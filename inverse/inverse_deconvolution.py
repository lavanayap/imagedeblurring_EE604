"""Validated 2D/3D circular inverse deconvolution for centered PSFs.

Modes
-----
naive     : G / H (tiny epsilon only). Baseline that shows noise explosion.
truncated : G / H where |H| >= threshold * max|H|, else 0 (pseudo-inverse).
clamped   : conj(H) / max(|H|^2, (threshold * max|H|)^2) (smooth floor).

Notes
-----
* The white-noise gain refers to the *linear, unclipped* inverse filter.
  Positivity clipping changes the effective noise statistics.
* threshold_for_noise_gain() applies to mode='truncated' only.
* Boundary handling is circular (FFT), same as the Wiener implementation.
"""
import numpy as np
from scipy.fft import rfftn, irfftn, ifftshift


def _pad_psf_to_shape(psf, shape):
    """Place a centered, compact PSF into a same-shaped centered array."""
    if psf.ndim != len(shape) or any(p > s for p, s in zip(psf.shape, shape)):
        raise ValueError('PSF must match number of dimensions and fit inside image')
    if psf.shape == tuple(shape):
        return psf.copy()
    out = np.zeros(shape, dtype=psf.dtype)
    sl = tuple(slice(s // 2 - p // 2, s // 2 - p // 2 + p)
               for p, s in zip(psf.shape, shape))
    out[sl] = psf
    return out


def _rfft_weights(shape):
    """Weights converting half-spectrum (rfftn) sums to full-spectrum sums."""
    w = np.full(shape[-1] // 2 + 1, 2.0)
    w[0] = 1.0
    if shape[-1] % 2 == 0:
        w[-1] = 1.0
    return np.broadcast_to(w, tuple(shape[:-1]) + (len(w),))


def estimate_noise_sigma(volume):
    """Heuristic Gaussian-equivalent noise std via MAD of X differences.

    Not a Poisson-Gaussian noise-model fit; sharp structures may bias it high.
    """
    a = np.asarray(volume, dtype=np.float64)
    if a.ndim not in (2, 3) or a.shape[-1] < 2 or not np.isfinite(a).all():
        raise ValueError('Expected finite 2D/3D image with X dimension >=2')
    d = np.diff(a, axis=-1)
    return float(np.median(np.abs(d - np.median(d)))
                 / (0.6744897501960817 * np.sqrt(2)))


class InverseFilter:
    def __init__(self, psf, shape):
        shape = tuple(shape)
        if len(shape) not in (2, 3) or any(
                not isinstance(s, (int, np.integer)) or s < 2 for s in shape):
            raise ValueError('Shape must be 2D/3D with each dimension >=2')

        psf = np.asarray(psf, dtype=np.float64)
        if psf.ndim != len(shape) or any(p < 1 for p in psf.shape):
            raise ValueError('PSF dimensions must match image dimensions')
        if not np.isfinite(psf).all():
            raise ValueError('PSF must be finite')
        # Tolerate tiny negative values (processing/rounding residue only).
        if psf.min() < 0:
            if psf.min() >= -1e-3 * psf.max():
                psf = np.clip(psf, 0, None)
            else:
                raise ValueError('PSF has significant negative values')
        total = float(np.sum(psf, dtype=np.float64))
        if not np.isfinite(total) or total <= 0:
            raise ValueError('PSF must have a finite, positive sum')

        psf = _pad_psf_to_shape(psf / total, shape)
        self.shape = shape
        self.H = rfftn(ifftshift(psf))
        self.absH = np.abs(self.H)
        self.Hmax = float(self.absH.max())
        self._w = _rfft_weights(shape)
        self._w_sum = float(np.prod(shape))

        # Cache gain / kept fraction at unique |H| levels so tied magnitudes
        # always enter and leave the filter as one group.
        flat_abs = self.absH.ravel()
        order = np.argsort(-flat_abs)
        sorted_abs = flat_abs[order]
        sorted_w = np.asarray(self._w).ravel()[order]

        reciprocal = np.zeros_like(sorted_abs, dtype=np.float64)
        nz = sorted_abs > 0
        reciprocal[nz] = sorted_w[nz] / np.maximum(sorted_abs[nz], 1e-150) ** 2
        cum_gain = np.cumsum(reciprocal) / self._w_sum
        cum_weight = np.cumsum(sorted_w) / self._w_sum

        group_ends = np.r_[np.flatnonzero(np.diff(sorted_abs) != 0),
                           len(sorted_abs) - 1]
        level_desc = sorted_abs[group_ends]
        valid = level_desc > 0
        # Ascending levels, with matching gain / kept fraction.
        self._levels = level_desc[valid][::-1]
        self._gain_at_level = cum_gain[group_ends[valid]][::-1]
        self._kept_at_level = cum_weight[group_ends[valid]][::-1]

    @staticmethod
    def _check_threshold(threshold):
        if (not np.isscalar(threshold) or not np.isfinite(threshold)
                or not (0 < threshold <= 1)):
            raise ValueError('threshold must be finite in (0, 1]')
        return float(threshold)

    def kept_fraction(self, threshold):
        t = self._check_threshold(threshold) * self.Hmax
        i = np.searchsorted(self._levels, t, side='left')
        return float(self._kept_at_level[i]) if i < len(self._levels) else 0.0

    def noise_gain(self, threshold):
        t = self._check_threshold(threshold) * self.Hmax
        i = np.searchsorted(self._levels, t, side='left')
        return float(self._gain_at_level[i]) if i < len(self._levels) else 0.0

    def threshold_for_noise_gain(self, max_gain):
        """Smallest threshold whose truncated noise gain is <= max_gain.

        Returns None if even the strongest-frequency group exceeds the budget;
        callers must handle that case explicitly.
        """
        if (not np.isscalar(max_gain) or not np.isfinite(max_gain)
                or max_gain < 0):
            raise ValueError('max_gain must be finite and nonnegative')
        feasible = np.flatnonzero(self._gain_at_level <= max_gain + 1e-12)
        if not len(feasible):
            return None
        t = float(self._levels[feasible[0]] / self.Hmax)
        return min(1.0, max(t, np.finfo(float).tiny))

    def restore(self, observed, threshold=0.05, mode='truncated', clip=True):
        """Returns (restored float32 array, info dict).

        Negative observed values (e.g. from simulated Gaussian read noise)
        are allowed; with clip=True the output is clipped to >= 0.
        """
        y = np.asarray(observed, dtype=np.float64)
        if y.shape != self.shape or not np.isfinite(y).all():
            raise ValueError('Observed must have matching shape and finite values')
        if mode not in ('naive', 'truncated', 'clamped'):
            raise ValueError('mode must be naive / truncated / clamped')
        if not isinstance(clip, (bool, np.bool_)):
            raise ValueError('clip must be boolean')
        # threshold is unused by naive mode, so do not validate it there.
        threshold = 0.0 if mode == 'naive' else self._check_threshold(threshold)

        t = threshold * self.Hmax
        if mode == 'naive':
            inverse = 1.0 / (self.H + 1e-12)
            kept = 1.0
        elif mode == 'truncated':
            keep = (self.absH >= t) & (self.absH > 0)
            inverse = np.zeros_like(self.H)
            inverse[keep] = 1.0 / self.H[keep]
            kept = self.kept_fraction(threshold)
        else:
            inverse = np.conj(self.H) / np.maximum(self.absH ** 2, t ** 2)
            kept = self.kept_fraction(threshold)

        gain = float(np.sum(self._w * (np.abs(inverse) ** 2),
                            dtype=np.float64) / self._w_sum)
        out = irfftn(rfftn(y) * inverse, s=self.shape)
        if clip:
            out = np.maximum(out, 0)
        if not np.isfinite(out).all():
            raise FloatingPointError('Nonfinite reconstruction')
        return out.astype(np.float32), {
            'mode': mode, 'threshold': threshold,
            'kept_fraction': float(kept), 'noise_gain': gain,
            'noise_gain_model': 'white Gaussian, linear unclipped output',
        }


def inverse_3d(observed, psf, threshold=0.05, mode='truncated'):
    restored, _ = InverseFilter(psf, np.shape(observed)).restore(
        observed, threshold, mode)
    return restored


def selftest():
    rng = np.random.default_rng(6)
    for shape in [(16, 16), (12, 12, 12)]:
        x = rng.random(shape, dtype=np.float32)
        psf = np.zeros(shape)
        psf[tuple(s // 2 for s in shape)] = 1
        restored, _ = InverseFilter(psf, shape).restore(x, threshold=0.05)
        np.testing.assert_allclose(restored, x, atol=1e-6)
        print('PASS identity PSF', shape)

    shape = (16, 16, 16)
    zz, yy, xx = np.meshgrid(*[np.arange(s) - s // 2 for s in shape],
                             indexing='ij')
    psf = np.exp(-(zz ** 2 + yy ** 2 + xx ** 2) / (2 * 1.1 ** 2))
    filt = InverseFilter(psf, shape)
    for b in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
        t = filt.threshold_for_noise_gain(b)
        if t is not None:
            assert filt.noise_gain(t) <= b + 1e-9, (t, b, filt.noise_gain(t))
    for t in [0.01, 0.05, 0.1, 0.5]:
        out, info = filt.restore(rng.random(shape), threshold=t)
        assert out.shape == shape and np.isfinite(out).all()
        assert abs(info['noise_gain'] - filt.noise_gain(t)) < 1e-8
    assert filt.threshold_for_noise_gain(0) is None

    # naive mode must accept threshold=0 and negative (read-noise) inputs
    filt.restore(rng.normal(size=shape), threshold=0.0, mode='naive')

    for bad_psf in (np.zeros(shape), np.full(shape, np.nan)):
        try:
            InverseFilter(bad_psf, shape)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid PSF accepted')
    print('PASS budgets, gains, shapes, naive/negatives, invalid inputs')


if __name__ == '__main__':
    selftest()