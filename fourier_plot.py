import numpy as np
import torch 
import matplotlib.pyplot as plt

# --- grid ---
N = 512
t = np.linspace(0, 2*np.pi, N, endpoint=False)

# --- 1) single sine wave ---
f = 7
x = np.sin(f * t)
X = np.fft.fftshift(np.fft.fft(x))
P = np.abs(X)

# --- 2) white-noise GRF (flat spectrum) ---
grf_white = np.random.normal(size=N)
G_white = np.fft.fftshift(np.fft.fft(grf_white))
P_white = np.abs(G_white)

# --- 3) GRF with k^-2 spectrum ---
k = np.fft.fftshift(np.fft.fftfreq(N, d=(t[1]-t[0])))

# power spectrum ~ 1/k^2, avoid k=0
P_target = np.where(k == 0, 0, 1.0 / (k**2))

# complex Gaussian noise
noise = np.random.normal(size=N) + 1j*np.random.normal(size=N)

# impose spectrum
G_red = np.sqrt(P_target) * noise
G_red_unshift = np.fft.ifftshift(G_red)
grf_red = np.fft.ifft(G_red_unshift).real

# its spectrum
G_red_disp = np.fft.fftshift(np.fft.fft(grf_red))
P_red = np.abs(G_red_disp)

# --- plot ---
plt.figure(figsize=(12,12))

# sine
plt.subplot(3,2,1)
plt.title("Sine wave, freq = {}".format(f))
plt.plot(t, x)
plt.xlim(0, 2*np.pi)

plt.subplot(3,2,2)
plt.title("Spectrum (double spike)")
plt.plot(k, P)

# white noise
plt.subplot(3,2,3)
plt.title("White-noise")
plt.plot(t, grf_white)
plt.xlim(0, 2*np.pi)

plt.subplot(3,2,4)
plt.title("Spectrum (flat)")
plt.plot(k, P_white)

# red noise
plt.subplot(3,2,5)
plt.title("GRF with $k^{-2}$ power")
plt.plot(t, grf_red)
plt.xlim(0, 2*np.pi)

plt.subplot(3,2,6)
plt.title("Spectrum (~$k^{-2}$)")
plt.plot(k, P_red)

plt.tight_layout()
plt.show()
