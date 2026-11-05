
import numpy as np
from scipy.fft import rfftn, irfftn, ifftshift


def richardson_lucy_calibrated_3d(
    observed,
    psf,
    alpha,
    beta,
    iterations=30,
    epsilon=1e-8
):
    """
    3D Richardson-Lucy with multiplicative gain
    and constant additive background.

    Observation model:
        observed = alpha * A(estimate) + beta

    Assumes a Poisson observation likelihood.
    """

    observed = np.asarray(observed, dtype=np.float32)
    psf = np.asarray(psf, dtype=np.float32)

    if observed.ndim != 3 or psf.ndim != 3:
        raise ValueError("Observed image and PSF must be 3D.")

    if observed.shape != psf.shape:
        raise ValueError("Image and PSF shapes must match.")

    if not np.all(np.isfinite(observed)):
        raise ValueError("Observed image contains invalid values.")

    if not np.all(np.isfinite(psf)):
        raise ValueError("PSF contains invalid values.")

    if np.any(observed < 0):
        raise ValueError("Observed image must be nonnegative.")

    if np.any(psf < 0) or psf.sum() <= 0:
        raise ValueError("PSF must be nonnegative with positive sum.")

    if alpha <= 0 or beta < 0:
        raise ValueError("alpha must be positive and beta nonnegative.")

    if iterations < 1 or epsilon <= 0:
        raise ValueError("Invalid iterations or epsilon.")

    # Normalize PSF
    psf = psf / psf.sum()

    # FFT representation
    H = rfftn(ifftshift(psf))
    H_conjugate = np.conjugate(H)

    def forward(volume):
        return irfftn(
            H * rfftn(volume),
            s=observed.shape
        ).real

    def adjoint(volume):
        return irfftn(
            H_conjugate * rfftn(volume),
            s=observed.shape
        ).real

    # Adjoint normalization
    normalization = np.maximum(
        adjoint(np.ones_like(observed)),
        epsilon
    )

    # Initial estimate in reference intensity scale
    estimate = np.maximum(
        observed - beta,
        epsilon
    ) / alpha

    # RL iterations
    for i in range(iterations):

        predicted = alpha * forward(estimate) + beta

        ratio = observed / np.maximum(
            predicted,
            epsilon
        )

        correction = adjoint(ratio) / normalization

        estimate *= np.maximum(correction, 0)

        estimate = np.maximum(estimate, 0)

        if (i + 1) % 10 == 0:
            print(
                f"Calibrated RL iteration "
                f"{i + 1}/{iterations} completed"
            )

    return estimate.astype(np.float32)
