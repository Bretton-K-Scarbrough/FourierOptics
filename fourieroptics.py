import numpy as np
import numpy.fft as fft
from typing import Tuple
from numpy.fft import fftshift, ifftshift, fft2
import matplotlib.pyplot as plt

PI = np.pi


def rect(t: np.ndarray, T: float = 1.0, t0: float = 0.0) -> np.ndarray:
    """
    Creates a 1D rect function

    Parameters:
    t         : linspace vector
    T         : width of rect function
    t0        : offset of rect function
    """
    return np.where(np.abs(t - t0) <= T / 2, 1, 0)


def rect2D(
    X: np.ndarray, Y: np.ndarray, wx: float, wy: float, amplitude: float = 1
) -> np.ndarray:
    """
    Creates a 2D rect function, intended to represent the transmittance function of a square aperture.

    Parameters:
    x         : x-axis of created mesh grid
    y         : y-axis of created mesh grid
    wx        : full-width of aperture in x direction in meters
    wy        : full-width of aperture in y idrection in meters
    amplitude : amplitude of transmittance function, defaults to 1.
    """
    # xx, yy = np.meshgrid(x, y)
    mask_x = np.abs(X) <= wx / 2
    mask_y = np.abs(Y) <= wy / 2
    unit_plane = amplitude * (mask_x & mask_y)
    return unit_plane


def circ(
    X: np.ndarray,
    Y: np.ndarray,
    r: float = 1.0,
    amplitude: float = 1.0,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """
    Creates a circ function, intended to represent the transmittance function of a circular aperture.

    Parameters:
    x         : x-axis of created mesh grid
    y         : y-axis of created mesh grid
    r         : radius of circular aperture
    amplitude : amplitude of transmittance function, defaults to 1.
    x0        : offset in x direction
    y0        : offset in y direction
    """
    distance = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    circle_mask = distance <= r
    return amplitude * circle_mask


def propFF(u1: np.ndarray, L1: float, lam: float, z: float) -> Tuple[np.ndarray, float]:
    """
    Propagates some field, u1, to z using the Fraunhofer kernel. Note: this is the same as propagating
    to far field or through a lens. Assumes a monochromatic field distribution

    Parameters:
    u1        : Input field distribution
    L1        : Physical length of u1
    lam       : wavelength of field
    z         : propagation distance (or focal length)

    Returns:
    Tuple[np.ndarray, float]
        u2 : np.ndarray
            Propagated field in the observation plane.
        L2 : float
            Physical length in the observation plane.
    """
    M = np.shape(u1)[0]
    dx1 = L1 / M
    k = 2 * PI / lam

    L2 = lam * z / dx1  # side length in obs plane
    x2 = np.linspace(-L2 / 2, L2 / 2, M)

    X2, Y2 = np.meshgrid(x2, x2)

    c = 1 / (1j * lam * z) * np.exp(1j * k / (2 * z) * (X2**2 + Y2**2))
    u2 = c * ifftshift(fft2(fftshift(u1))) * dx1**2

    return u2, L2


def holoeye_SLM(phasemask: np.ndarray, L: float, pixel_subdivision: int) -> np.ndarray:
    SLM_x = 15.36e-3
    SLM_y = 8.64e-3
    pixel_pitch = 8e-6
    fill_factor = 0.93
    actual_pixel_length = pixel_pitch * fill_factor

    dx = pixel_pitch / pixel_subdivision
    N = int(L / dx)

    x = np.linspace(-L / 2, L / 2, N)
    X, Y = np.meshgrid(x, x)

    # NOTE: Create SLM pixel array
    samples_per_pixel = int(pixel_pitch / dx)
    t = np.linspace(-pixel_pitch / 2, pixel_pitch / 2, samples_per_pixel)
    single_pixel_1D = rect(t, actual_pixel_length)
    single_pixel_2D = np.tile(single_pixel_1D, (samples_per_pixel, 1))
    single_pixel_2D = single_pixel_2D * single_pixel_2D.T
    full_pixel_array = np.tile(single_pixel_2D, (1080, 1920))

    upscaled_phase_mask = np.repeat(
        np.repeat(phasemask, samples_per_pixel, axis=0), samples_per_pixel, axis=1
    )
    # NOTE: Create u1
    u1 = full_pixel_array * np.exp(1j * upscaled_phase_mask)

    # NOTE: Put SLM in full viewing window
    height_slm, width_slm = np.shape(full_pixel_array)
    height_full, width_full = np.shape(X)

    start_x = (width_full - width_slm) // 2
    start_y = (height_full - height_slm) // 2

    # NOTE: Paste SLM array into full viewing window
    SLM_in_space = np.zeros(np.shape(X), dtype=complex)
    SLM_in_space[start_y : start_y + height_slm, start_x : start_x + width_slm] = u1
    return SLM_in_space
