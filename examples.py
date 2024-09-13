import numpy as np
import matplotlib.pyplot as plt
from fourieroptics.propagators import propTF_F, propTF_RS, propIR_RS, propIR_F
from fourieroptics.apertures import rect2D

L = 0.5
N = int(0.5e3)
dx = L / N

x = np.linspace(-L / 2, L / 2 - dx, N)
X, Y = np.meshgrid(x, x)

lam = 0.5e-6
k = 2 * np.pi / lam
w = 0.051
z = 100

u1 = rect2D(X, Y, 2 * w, 2 * w)
I1 = np.abs(u1**2)

plt.figure()
extent = (-L / 2, L / 2, -L / 2, L / 2)
plt.imshow(I1, cmap="jet", extent=extent)
plt.title("Aperture, z = 0")

u2 = propTF_F(u1, L, lam, z)
I2 = np.abs(u2**2)

plt.figure()
extent = (-L / 2, L / 2, -L / 2, L / 2)
plt.imshow(I2, cmap="jet", extent=extent)
plt.title(f"U2 Fresnel TF Approach, z = {z}")

# NOTE: Assumptions for use of this plot are validated. Therefore, this will not yield correct results, look at TF approach for valid results.
u3 = propIR_F(u1, L, lam, z)
I3 = np.abs(u3**2)

plt.figure()
extent = (-L / 2, L / 2, -L / 2, L / 2)
plt.imshow(I3, cmap="jet", extent=extent)
plt.title(f"U2 Fresnel IR Approach, z = {z}")

u4 = propTF_RS(u1, L, lam, z)
I4 = np.abs(u4**2)

plt.figure()
extent = (-L / 2, L / 2, -L / 2, L / 2)
plt.imshow(I4, cmap="jet", extent=extent)
plt.title(f"U2 RS TF Approach, z = {z}")

# NOTE: Assumptions for use of this plot are validated. Therefore, this will not yield correct results, look at TF approach for valid results.
u5 = propIR_RS(u1, L, lam, z)
I5 = np.abs(u5**2)

plt.figure()
extent = (-L / 2, L / 2, -L / 2, L / 2)
plt.imshow(I5, cmap="jet", extent=extent)
plt.title(f"U2 RS IR Approach, z = {z}")

# plt.figure()
# plt.plot(x, np.abs(u2[int(N / 2)]))
# plt.title(f"U2 Fresnel TF Approach Mag, z = {z}")
#
# plt.figure()
# plt.plot(x, np.unwrap(np.angle(u2[int(N / 2)])))
# plt.title(f"U2 Fresnel TF Approach Phase, z = {z}")
plt.show()
