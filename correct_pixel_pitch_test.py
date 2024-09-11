import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft, ifft, ifftshift
import fourieroptics as fo
from engineering_notation import eng_notation


def comb(x):
    x = np.round(x * 10**6) / 10**6
    out = (np.remainder(x, 1) == 0).astype(int)
    return out


def rect(t, T=1.0, t0=0.0):
    return np.where(np.abs(t - t0) <= T / 2, 1, 0)


lam = 532e-9

SLM_x = 15.36e-3
SLM_y = 8.64e-3
pixel_pitch = 8e-6
fill_factor = 0.93
actual_pixel_length = pixel_pitch * fill_factor
print(f"actual pixel length {actual_pixel_length}")
print(f"actual pixel length {actual_pixel_length/2}")

L = SLM_x * 2
dx = pixel_pitch / 4
N = int(L / dx)
print(f"N = {N}")

x = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, x)
print(f"shape of X = {np.shape(X)}")

samples_per_pixel = int(pixel_pitch / dx)

t = np.linspace(-pixel_pitch / 2, pixel_pitch / 2, samples_per_pixel)
single_pixel_1D = rect(t, actual_pixel_length)

single_pixel_2D = np.tile(single_pixel_1D, (samples_per_pixel, 1))
single_pixel_2D = single_pixel_2D * single_pixel_2D.T
print(f"shape of 2D pixel array = {np.shape(single_pixel_2D)}")

plt.figure()
plt.imshow(
    single_pixel_2D,
    cmap="gray",
)

full_pixel_array = np.tile(single_pixel_2D, (1080, 1920))
plt.figure()
plt.imshow(
    full_pixel_array,
    cmap="gray",
)
np.save("full_pixel_array.npy", full_pixel_array)
# print(f"shape of 2D pixel array = {np.shape(full_pixel_array)}")

phase_mask = np.load("./gerchberg_saxton/PenrosePhasemask.npy")
phase_mask = np.mod(phase_mask, 2 * np.pi)
# block = np.ones(samples_per_pixel)
# phase_mask = np.kron(phase_mask, block)
upscaled_phase_mask = np.repeat(
    np.repeat(phase_mask, samples_per_pixel, axis=0), samples_per_pixel, axis=1
)
print(f"Shape of upscaled phase mask {np.shape(upscaled_phase_mask)}")
plt.imshow(phase_mask, cmap="gray")
plt.show()
# exit()

full_pixel_array = np.load("./full_pixel_array.npy")

u1 = full_pixel_array * np.exp(1j * upscaled_phase_mask)
print(f"shape of u1 = {np.shape(u1)}")

# plt.imshow(np.angle(u1), cmap="gray")
# plt.show()
# plt.colorbar()
# exit()

height_slm, width_slm = np.shape(full_pixel_array)
height_full, width_full = np.shape(X)

start_x = (width_full - width_slm) // 2
start_y = (height_full - height_slm) // 2

# NOTE: Paste SLM array into full viewing window
SLM_in_space = np.zeros(np.shape(X), dtype=complex)
SLM_in_space[start_y : start_y + height_slm, start_x : start_x + width_slm] = u1
# plt.figure()
# plt.imshow(np.abs(SLM_in_space), cmap="gray")
# plt.figure()
# plt.imshow(np.angle(SLM_in_space), cmap="gray")
# plt.show()

u2, L2 = fo.propFF(SLM_in_space, L, lam, 50e-2)
np.save("u2.npy", u2)
print(f"L2 = {eng_notation(L2)}")
print("YIPPPEEEEE!")
exit()
I2 = np.abs(u2**2)
plt.imshow(I2, cmap="gray")
plt.show()
