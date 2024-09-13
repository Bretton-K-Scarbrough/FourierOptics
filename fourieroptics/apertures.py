import numpy as np


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
