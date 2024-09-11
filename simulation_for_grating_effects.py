import numpy as np
import matplotlib.pyplot as plt
import fourieroptics as fo
from numpy.fft import fftshift, fft, ifftshift, ifft


def comb(x):
    x = np.round(x * 10**6) / 10**6
    out = (np.remainder(x, 1) == 0).astype(int)
    return out


def rect(t, T=1.0, t0=0.0):
    return np.where(np.abs(t - t0) <= T / 2, 1, 0)


lam = 532e-9

Lx = 15.36e-3
Ly = Lx
dx = 4e-6
dy = dx
Nx = int(Lx / dx)
Ny = Nx

x = np.linspace(-Lx / 2, Lx / 2 - dx, Nx)
X, Y = np.meshgrid(x, x)


# NOTE: Square Aperture Creation
u1 = np.ones(np.shape(X))  # Creates a uniform plane wave the size of the viewing area

wx = 5e-3
wy = 3e-3
aperture = fo.rect2D(X, Y, wx, wy)
u1 = aperture

# NOTE: Grating Function Creation
P = 2 * 8e-6  # grating period
fc = fft(fftshift(comb(x / P)))
fr = fft(fftshift(rect(x / (P / 2))))
ux = ifftshift(ifft(fc * fr))  # 1D convolution of rect and comb
u_grating = np.tile(ux, (Nx, 1))
u_grating = u_grating * u_grating.T
# u_grating = u_grating * rect(X / wx) * rect(Y / wy)
u_grating = u_grating * aperture
u1 = u1 * u_grating
I_grating = np.abs(u1**2)
# plt.imshow(I_grating, cmap="jet", extent=(-Lx / 2, Lx / 2, -Lx / 2, Lx / 2))
# plt.show()
print(f"Shape of u1: {np.shape(u1)}")


# NOTE: Spatial Filtering
iris = fo.circ(X, Y, r=2e-3, x0=5e-3)

# NOTE: Lens Creation
f = 1
u2, L2x = fo.propFF(u1, Lx, lam, f)
u2_filtered = u2 * iris
u3, L3x = fo.propFF(u2_filtered, L2x, lam, f)
I1 = np.abs(u1**2)
I2 = np.abs((u2) ** 2)
I3 = np.abs(u3**2)

# NOTE: Plotting
plt.figure()
Lx_mm = Lx * 1e3
plt.imshow(I1, cmap="jet", extent=(-Lx_mm / 2, Lx_mm / 2, -Lx_mm / 2, Lx_mm / 2))
plt.title("SLM Screen")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")

plt.figure()
L2x_mm = L2x * 1e3
plt.imshow(
    np.cbrt(I2), cmap="jet", extent=(-L2x_mm / 2, L2x_mm / 2, -L2x_mm / 2, L2x_mm / 2)
)
plt.title("L1 cbrt(I2)")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")

plt.figure()
plt.imshow(iris, cmap="jet", extent=(-L2x_mm / 2, L2x_mm / 2, -L2x_mm / 2, L2x_mm / 2))
plt.title("Spatial Filter")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")


plt.figure()
L3x_mm = L3x * 1e3
plt.imshow(I3, cmap="jet", extent=(-L3x_mm / 2, L3x_mm / 2, -L3x_mm / 2, L3x_mm / 2))
plt.title("L2")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.show()
