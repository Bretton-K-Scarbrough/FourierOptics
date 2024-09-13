# FourierOptics
*fourieroptics* is a Python library designed to streamline the simulation of optical setups in the Fourier domain. It provides various methods to simulate wave propagation, supports common apertures, and includes helper functions to determine which propagation methods are valid under different conditions.

The library is useful for researchers and engineers working in fields such as optics, computational imaging, and photonics, allowing them to quickly set up and analyze optical wave propagation problems.

## Features
- Wave Propagation: Simulates wave propagation using different methods, such as Rayleigh-Sommerfeld, Fresnel, and Fraunhofer approximations.
- Aperture Functions: Common aperture shapes like circular and square are provided to easily model optical systems.
- Helper Functions: Tools to determine the validity of propagation methods and other utility functions.

# Installation
To install this library run the command below in the shell
```bash
pip install fourieroptics
```

# Usage

To utilize the library import the propagators module
```Python
import fourieroptics as fo
import fourieroptics.propagators # Contains propagating methods
import fourieroptics.apertures # Contains simple common apertures
import fourier.optics.helpers # Contains minor reference materials
```

This library serves as a method to quickly propagate a given wave, u1, located at z = z1, to u2, located at z = z2. It can do this through several different methods, including different regimes.

## Simple Square Aperture Example

```Python
import numpy as np
import matplotlib.pyplot as plt
import fourieroptics.propagators as prop
import fourieroptics.apertures as aperture

L = 0.5  # Observation window length in m (always assume square windows)
N = 2000  # Number of samples created
# NOTE: Large N can cause dramatically longer run times or even crashes
dx = L / N

x = np.linspace(-L / 2, L / 2 - dx, N)
X, Y = np.meshgrid(x, x)  # As we assume a square window we don't  "y"

lam = 0.5e-6  # Wavelength of light
w = 2 * 0.051  # Full width of square aperture

# Creates square aperture, with uniform plane wave illumination
u1 = aperture.rect2D(X, Y, w, w)
I1 = np.abs(u1**2)
plt.figure()
plt.imshow(I1, cmap="jet", extent=(-L / 2, L / 2, -L / 2, L / 2))
plt.title(f"I1 z = 0m")
plt.suptitle(f"Aperture Illuminated via Uniform Plane Wave")
plt.xlabel("x (m)")
plt.ylabel("y (m)")

z = 30  # distance to propagate in m

# Propagates u1 via the Rayleigh-Sommerfeld kernel, using the TF approach
u2 = prop.propTF_RS(u1, L, lam, z)
I2 = np.abs(u2**2)
plt.figure()
plt.imshow(I2, cmap="jet", extent=(-L / 2, L / 2, -L / 2, L / 2))
plt.xlim(-0.075, 0.075)
plt.ylim(-0.075, 0.075)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title(f"I2 (zoomed in) at z = {z}m")
plt.suptitle(f"Propagated via Rayleigh-Sommerfeld Kernel")

plt.show()
```
## Calling Helper Functions
The code below will call a helper function which will print the general design steps to creating a propagation simulation.
```Python
import fouieroptics.helpers
fourieroptics.helpers.design_steps()
```
