import numpy as np
from typing import Tuple
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


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
    Propagates some field, u1, to z using the Fraunhofer kernel. Note: this is the same as propagating to far field or through a lens. Assumes a monochromatic field distribution

    Parameters:
    u1        : np.ndarray (M,M), complex|real
                2D complex|real array Input field distribution
    L1        : Physical length of u1
    lam       : wavelength of field
    z         : propagation distance (or focal length)

    Returns:
    Tuple[np.ndarray, float]
    Returns:
        u2 : np.ndarray (M,M), complex
             2D complex array Propagated field in the observation plane.
        L2 : float
             Physical length in the observation plane.
    """
    assert u1.ndim == 2, "Input field u1 must be a 2D array"
    assert L1 > 0, "Physical length L must be positive"
    assert lam > 0, "Wavelength lam must be positive"
    assert z != 0, "Propagation distance z cannot be zero"

    M = np.shape(u1)[0]
    dx1 = L1 / M
    k = 2 * np.pi / lam

    L2 = lam * z / dx1  # side length in obs plane
    x2 = np.linspace(-L2 / 2, L2 / 2, M)

    X2, Y2 = np.meshgrid(x2, x2)

    c = 1 / (1j * lam * z) * np.exp(1j * k / (2 * z) * (X2**2 + Y2**2))
    u2 = c * ifftshift(fft2(fftshift(u1))) * dx1**2

    return u2, L2


def propTF_F(u1: np.ndarray, L: float, lam: float, z: float) -> np.ndarray:
    """
    Propagates some field, u1, to z using the Fresnel kernel transfer function approach.

    #WARNING: Before calling this function it is important to check if the Fresnel criterion has been met as well as if you should be using the TF approach or the IR approach.

    Parameters:
    u1        : np.ndarray (M,M), complex|real
                2D complex|real array Input field distribution
    L1        : Physical length of u1
    lam       : wavelength of field
    z         : propagation distance

    Returns:
        u2 : np.ndarray (M,M), complex
             2D complex array Propagated field in the observation plane.
    """

    assert u1.ndim == 2, "Input field u1 must be a 2D array"
    assert L > 0, "Physical length L must be positive"
    assert lam > 0, "Wavelength lam must be positive"
    assert z != 0, "Propagation distance z cannot be zero"

    M = u1.shape[0]
    dx = L / M

    # Creates frequency space coords
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, M)
    FX, FY = np.meshgrid(fx, fx)

    # Transfer function
    H = np.exp(-1j * np.pi * lam * z * (FX**2 + FY**2))
    H = np.fft.fftshift(H)

    U1 = np.fft.fft2(np.fft.fftshift(u1))

    # Multiply in frequency domain
    U2 = H * U1

    # Inverse FFT, center observation field
    u2 = np.fft.ifftshift(np.fft.ifft2(U2))
    return u2


def propIR_F(u1: np.ndarray, L: float, lam: float, z: float) -> np.ndarray:
    """
    Propagates some field, u1, to z using the Fresnel kernel impulse response approach.

    #WARNING: Before calling this function it is important to check if the Fresnel criterion has been met. If the Fresnel criterion has not been met, consider using the Rayleigh-Sommerfeld functions instead.

    It is important to check if the TF approach or IR approach is more accurate when calling this function. Look at trade of table in Chapter 5 of "Computational Fourier Optics" by David Voelz

    Parameters:
    u1        : np.ndarray (M,M), complex|real
                2D complex|real array Input field distribution
    L1        : Physical length of u1
    lam       : wavelength of field
    z         : propagation distance

    Returns:
        u2 : np.ndarray (M,M), complex
             2D complex array Propagated field in the observation plane.
    """

    assert u1.ndim == 2, "Input field u1 must be a 2D array"
    assert L > 0, "Physical length L must be positive"
    assert lam > 0, "Wavelength lam must be positive"
    assert z != 0, "Propagation distance z cannot be zero"

    M = np.shape(u1)[0]
    dx = L / M
    k = 2 * np.pi / lam
    x = np.linspace(-L / 2, L / 2, M)
    X, Y = np.meshgrid(x, x)
    h = (1 / (1j * lam * z)) * np.exp((1j * k) / (2 * z) * (X**2 + Y**2))
    H = fftshift(h)
    H = fft2(H) * dx**2

    U1 = fftshift(u1)
    U1 = fft2(U1)

    U2 = U1 * H
    u2 = ifft2(U2)
    u2 = ifftshift(u2)
    return u2


def propTF_RS(u1: np.ndarray, L: float, lam: float, z: float) -> np.ndarray:
    """
    Propagates some field, u1, to z using the Rayleigh-Sommerfled kernel, transfer function approach.

    #WARNING: It is important to check if the TF approach or IR approach is more accurate when calling this function. Look at trade of table in Chapter 5 of "Computational Fourier Optics" by David Voelz

    Parameters:
    u1        : np.ndarray (M,M), complex|real
                2D complex|real array Input field distribution
    L1        : Physical length of u1
    lam       : wavelength of field
    z         : propagation distance

    Returns:
        u2 : np.ndarray (M,M), complex
             2D complex array Propagated field in the observation plane.
    """

    assert u1.ndim == 2, "Input field u1 must be a 2D array"
    assert L > 0, "Physical length L must be positive"
    assert lam > 0, "Wavelength lam must be positive"
    assert z != 0, "Propagation distance z cannot be zero"

    M = u1.shape[0]  # Get input field array size
    dx = L / M  # Sample interval
    k = 2 * np.pi / lam

    # Frequency coordinates
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, M)
    FX, FY = np.meshgrid(fx, fx)

    # Transfer function
    H = np.exp(1j * k * z * np.sqrt(1 - (lam * FX) ** 2 - (lam * FY) ** 2))
    H = np.fft.fftshift(H)

    # FFT of the source field
    U1 = np.fft.fft2(np.fft.fftshift(u1))

    # Multiply in frequency domain
    U2 = H * U1

    # Inverse FFT, center observation field
    u2 = np.fft.ifftshift(np.fft.ifft2(U2))

    return u2


def propIR_RS(u1: np.ndarray, L: float, lam: float, z: float) -> np.ndarray:
    """
    Propagates some field, u1, to z using the Rayleigh-Sommerfled kernel, transfer function approach.

    #WARNING: It is important to check if the TF approach or IR approach is more accurate when calling this function. Look at trade of table in Chapter 5 of "Computational Fourier Optics" by David Voelz

    Parameters:
    u1        : np.ndarray (M,M), complex|real
                2D complex|real array Input field distribution
    L1        : Physical length of u1
    lam       : wavelength of field
    z         : propagation distance

    Returns:
        u2 : np.ndarray (M,M), complex
             2D complex array Propagated field in the observation plane.
    """
    assert u1.ndim == 2, "Input field u1 must be a 2D array"
    assert L > 0, "Physical length L must be positive"
    assert lam > 0, "Wavelength lam must be positive"
    assert z != 0, "Propagation distance z cannot be zero"

    M = np.shape(u1)[0]
    dx = L / M
    x = np.linspace(-L / 2, L / 2 - dx, M)
    k = 2 * np.pi / lam
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2 + z**2)
    h = z / (1j * lam * r**2) * np.exp(1j * k * r)
    H = fftshift(h)
    H = fft2(H) * dx**2

    U1 = fftshift(u1)
    U1 = fft2(U1)

    U2 = U1 * H
    u2 = ifft2(U2)
    u2 = ifftshift(u2)
    return u2
