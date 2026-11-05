
import numpy as np
from scipy.fft import rfftn, irfftn, ifftshift


def richardson_lucy_3d(
    observed,
    psf,
    iterations=30,
    epsilon=1e-8
):
    """
    3D Richardson-Lucy Deconvolution.

    Parameters
    ----------
    observed : 3D numpy array
        Blurred and noisy microscopy volume.

    psf : 3D numpy array
        Point Spread Function.

    iterations : int
        Number of RL iterations.

    epsilon : float
        Small constant to prevent division by zero.

    Returns
    -------
    restored : 3D numpy array
        Restored microscopy volume.
    """

    # 1. Convert inputs to float32
    observed = np.asarray(observed, dtype=np.float32)
    psf = np.asarray(psf, dtype=np.float32)

    # 2. Validate dimensions
    if observed.ndim != 3 or psf.ndim != 3:
        raise ValueError("Image and PSF must both be 3D.")

    if observed.shape != psf.shape:
        raise ValueError("Image and PSF must have identical shapes.")

    if iterations < 1:
        raise ValueError("Iterations must be at least 1.")

    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")

    if not np.all(np.isfinite(observed)):
        raise ValueError("Observed image contains invalid values.")

    if not np.all(np.isfinite(psf)):
        raise ValueError("PSF contains invalid values.")

    if np.any(psf < 0) or psf.sum() <= 0:
        raise ValueError("PSF must be nonnegative with positive sum.")

    # 3. Ensure nonnegative observations
    observed = np.maximum(observed, 0)

    # 4. Normalize PSF
    psf = psf / np.sum(psf)

    # 5. Move centered PSF to FFT origin
    psf_origin = ifftshift(psf)

    # 6. Calculate Fourier transform of PSF
    H = rfftn(psf_origin)

    # Complex conjugate represents the adjoint operation
    H_conjugate = np.conjugate(H)

    # 7. Initialize estimated image
    estimate = np.maximum(observed.copy(), epsilon)

    # 8. Richardson-Lucy iterations
    for i in range(iterations):

        # Forward convolution
        blurred_estimate = irfftn(
            H * rfftn(estimate),
            s=observed.shape
        ).real

        # Calculate ratio
        ratio = observed / np.maximum(
            blurred_estimate, epsilon
        )

        # Back-project correction
        correction = irfftn(
            H_conjugate * rfftn(ratio),
            s=observed.shape
        ).real

        # Update estimate
        estimate *= correction

        # Maintain nonnegative intensity
        estimate = np.maximum(estimate, 0)

        if (i + 1) % 10 == 0:
            print(
                f"RL iteration {i + 1}/{iterations} completed"
            )

    return estimate.astype(np.float32)
