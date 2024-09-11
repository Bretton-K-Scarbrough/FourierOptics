import numpy as np
import matplotlib.pyplot as plt
import fourieroptics as fo

L = 0.5
N = int(0.5e3)
dx = L / N

x = np.linspace(-L / 2, L / 2 - dx, N)
X, Y = np.meshgrid(x, x)

lam = 0.5e-6
k = 2 * np.pi / lam
w = 0.051
z = 10

u1 = fo.rect2D(X, Y, 2 * w, 2 * w)
I1 = np.abs(u1**2)

plt.figure()
extent = (-L / 2, L / 2, -L / 2, L / 2)
plt.imshow(I1, cmap="jet", extent=extent)
plt.title("Aperture, z = 0")

u2 = fo.propTF_F(u1, L, lam, z)
I2 = np.abs(u2**2)

plt.figure()
extent = (-L / 2, L / 2, -L / 2, L / 2)
plt.imshow(I2, cmap="jet", extent=extent)
plt.title(f"U2 Fresnel TF Approach, z = {z}")

u3 = fo.propIR_F(u1, L, lam, z)
I3 = np.abs(u3**2)

plt.figure()
extent = (-L / 2, L / 2, -L / 2, L / 2)
plt.imshow(I3, cmap="jet", extent=extent)
plt.title(f"U2 Fresnel IR Approach, z = {z}")

u4 = fo.propTF_RS(u1, L, lam, z)
I4 = np.abs(u4**2)

plt.figure()
extent = (-L / 2, L / 2, -L / 2, L / 2)
plt.imshow(I4, cmap="jet", extent=extent)
plt.title(f"U2 RS TF Approach, z = {z}")

u5 = fo.propIR_RS(u1, L, lam, z)
I5 = np.abs(u5**2)

plt.figure()
extent = (-L / 2, L / 2, -L / 2, L / 2)
plt.imshow(I5, cmap="jet", extent=extent)
plt.title(f"U2 RS IR Approach, z = {z}")

# plt.figure()
# plt.plot(x, np.abs(u2[int(N / 2)]))
# plt.title("U2 Fresnel TF Approach Mag, z = 2000")
#
# plt.figure()
# plt.plot(x, np.unwrap(np.angle(u2[int(N / 2)])))
# plt.title("U2 Fresnel TF Approach Phase, z = 2000")
plt.show()
