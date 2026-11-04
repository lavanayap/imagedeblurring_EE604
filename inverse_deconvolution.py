
import numpy as np
from scipy.fft import fft2, ifft2


def psf_to_otf(psf, shape):
    """
    Convert a 2D Point Spread Function (PSF)
    into its Optical Transfer Function (OTF).

    Parameters
    ----------
    psf : ndarray
        2D PSF kernel.
    shape : tuple
        Desired output dimensions (height, width).

    Returns
    -------
    otf : ndarray
        Complex-valued OTF in the frequency domain.
    """

    psf = np.asarray(psf, dtype=np.float64)

    # Validate PSF
    if psf.ndim != 2:
        raise ValueError("PSF must be a 2D array.")

    if not np.all(np.isfinite(psf)):
        raise ValueError("PSF contains NaN or Inf.")

    if np.any(psf < 0):
        raise ValueError("PSF must be nonnegative.")

    if psf.sum() <= 0:
        raise ValueError("PSF must have a positive sum.")

    if len(shape) != 2:
        raise ValueError("Image shape must contain two dimensions.")

    if any(not isinstance(s, (int, np.integer)) or s <= 0 for s in shape):
        raise ValueError("Image dimensions must be positive integers.")

    kh, kw = psf.shape
    height, width = shape

    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("PSF dimensions must be odd.")

    if kh > height or kw > width:
        raise ValueError("PSF cannot be larger than the image.")

    # Normalize PSF
    psf = psf / psf.sum()

    # Embed PSF in an image-sized array
    padded_psf = np.zeros(shape, dtype=np.float64)
    padded_psf[:kh, :kw] = psf

    # Move PSF center to origin
    padded_psf = np.roll(
        padded_psf,
        shift=(-(kh // 2), -(kw // 2)),
        axis=(0, 1)
    )

    # Convert to frequency domain
    otf = fft2(padded_psf)

    return otf


def inverse_filter(
    blurred_image,
    psf,
    relative_floor=1e-3,
    clip_nonnegative=False
):
    """
    Stabilized inverse filtering in the Fourier domain.

    Parameters
    ----------
    blurred_image : ndarray
        2D grayscale degraded image.

    psf : ndarray
        2D point spread function.

    relative_floor : float
        Relative OTF magnitude threshold.
        Frequencies below this threshold are discarded.
        Must lie strictly between 0 and 1.

    clip_nonnegative : bool
        If True, negative reconstructed intensities
        are clipped to zero.

    Returns
    -------
    restored_image : ndarray
        Reconstructed 2D image.

    Notes
    -----
    The frequency threshold is externally configurable,
    allowing a future adaptive strategy to control it
    without modifying this function.
    """

    # Convert image to float
    image = np.asarray(blurred_image, dtype=np.float64)

    # Validate image
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale (2D).")

    if image.size == 0:
        raise ValueError("Input image cannot be empty.")

    if not np.all(np.isfinite(image)):
        raise ValueError("Input image contains NaN or Inf.")

    if not isinstance(relative_floor, (int, float)):
        raise TypeError("relative_floor must be a number.")

    if not np.isfinite(relative_floor) or not (0 < relative_floor < 1):
        raise ValueError("relative_floor must be between 0 and 1.")

    if not isinstance(clip_nonnegative, (bool, np.bool_)):
        raise TypeError("clip_nonnegative must be boolean.")

    # Validate PSF before accessing dimensions
    psf = np.asarray(psf, dtype=np.float64)

    if psf.ndim != 2:
        raise ValueError("PSF must be a 2D array.")

    if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
        raise ValueError("PSF dimensions must be odd.")

    # Padding dimensions
    kh, kw = psf.shape

    pad_h = kh // 2
    pad_w = kw // 2

    # Reflective padding
    padded_image = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="reflect"
    )

    # Fourier transform of degraded image
    G = fft2(padded_image)

    # Generate OTF
    H = psf_to_otf(psf, padded_image.shape)

    # Calculate OTF magnitude
    H_magnitude = np.abs(H)

    max_magnitude = np.max(H_magnitude)

    if max_magnitude <= 0:
        raise ValueError("Invalid PSF frequency response.")

    # Frequency threshold
    threshold = relative_floor * max_magnitude

    # Construct stabilized inverse filter
    H_inverse = np.zeros_like(H, dtype=np.complex128)

    valid = H_magnitude >= threshold

    H_inverse[valid] = 1.0 / H[valid]

    # Frequency-domain reconstruction
    F_estimated = G * H_inverse

    # Inverse Fourier transform
    restored = np.real(ifft2(F_estimated))

    # Remove padding
    restored_image = restored[
        pad_h:pad_h + image.shape[0],
        pad_w:pad_w + image.shape[1]
    ]

    # Optional nonnegative intensity constraint
    if clip_nonnegative:
        restored_image = np.maximum(restored_image, 0.0)

    return restored_image
