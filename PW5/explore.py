import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, spectrogram

d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z = d["Z"].to_numpy()
N = d["N"].to_numpy()
E = d["E"].to_numpy()
fs = 1.0 / (t[1] - t[0])
n_samples = len(t)

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
for ax, ch, lbl in zip(axs, [Z, N, E], ["Z", "N", "E"]):
    ax.plot(t, ch * 1000)
    ax.set_ylabel(f"{lbl} (mm/s)")
axs[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("step1_record.png")
plt.close()

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
m60 = t <= 60
for ax, ch, lbl in zip(axs, [Z, N, E], ["Z", "N", "E"]):
    ax.plot(t[m60], ch[m60] * 1000)
    ax.set_ylabel(f"{lbl} (mm/s)")
axs[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("step1_zoom60.png")
plt.close()

X_Z = np.fft.fft(Z)
freqs = np.fft.fftfreq(n_samples, d=1/fs)
amp_Z = np.abs(X_Z) / n_samples * 2

fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(amp_Z)
axs[0].set_xlabel("Array Index")
axs[0].set_ylabel("Amplitude")
axs[1].plot(freqs, amp_Z)
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Amplitude")
plt.tight_layout()
plt.savefig("step2a_fft.png")
plt.close()

Z_quiet = Z[m60]
n_q = len(Z_quiet)
f_q = np.fft.fftfreq(n_q, d=1/fs)
amp_q = np.abs(np.fft.fft(Z_quiet)) / n_q * 2

plt.figure(figsize=(8, 5))
plt.loglog(freqs[freqs > 0], amp_Z[freqs > 0], label="Whole record")
plt.loglog(f_q[f_q > 0], amp_q[f_q > 0], label="First 60 s")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("step2b_loglog.png")
plt.close()

assert np.allclose(np.fft.ifft(np.fft.fft(Z)).real, Z)

def bandpass(x, fs, a, b):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), d=1/fs)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0
    return np.fft.ifft(X).real

Z_clean = bandpass(Z, fs, 0.5, 10.0)
N_clean = bandpass(N, fs, 0.5, 10.0)
E_clean = bandpass(E, fs, 0.5, 10.0)

plt.figure(figsize=(10, 3))
plt.plot(t, (Z - Z_clean) * 1000)
plt.xlabel("Time (s)")
plt.ylabel("Removed noise (mm/s)")
plt.tight_layout()
plt.savefig("step3_removed.png")
plt.close()

def envelope(x, fs, win_s):
    w = max(1, int(round(win_s * fs)))
    return np.convolve(np.abs(x), np.ones(w) / w, mode="same")

H_clean = np.sqrt(N_clean**2 + E_clean**2)
env_Z = envelope(Z_clean, fs, 0.5)
env_H = envelope(H_clean, fs, 0.5)

plt.figure(figsize=(10, 4))
for win in [0.05, 0.5, 5.0]:
    plt.plot(t, envelope(Z_clean, fs, win) * 1000, label=f"{win} s")
plt.xlabel("Time (s)")
plt.ylabel("Envelope (mm/s)")
plt.legend()
plt.tight_layout()
plt.savefig("step4_envelopes.png")
plt.close()

idx_P = np.argmax(env_Z > 0.05 * np.max(env_Z))
t_P = t[idx_P]

mask_S = t > (t_P + 2.0)
idx_S_rel = np.argmax(env_H[mask_S] > 0.10 * np.max(env_H))
t_S = t[mask_S][idx_S_rel]

peaks, _ = find_peaks(env_H, prominence=np.max(env_H)*0.5)
t_max_H = t[peaks[np.argmax(env_H[peaks])]]

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
for ax, ch, lbl in zip(axs, [Z_clean, N_clean, E_clean], ["Z", "N", "E"]):
    ax.plot(t, ch * 1000)
    ax.axvline(t_P, color="g", linestyle="--", label="t_P")
    ax.axvline(t_S, color="r", linestyle="--", label="t_S")
    ax.axvline(t_max_H, color="k", linestyle=":", label="Max H")
    ax.set_ylabel(f"{lbl} (mm/s)")
axs[0].legend()
axs[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("step5_arrivals.png")
plt.close()

v_P = 6.2
v_S = 3.6
R = (t_S - t_P) / (1.0 / v_S - 1.0 / v_P)
dR_1s = 1.0 / (1.0 / v_S - 1.0 / v_P)
t_0 = t_P - R / v_P

f_spec, ts_spec, S_spec = spectrogram(Z, fs=fs, nperseg=256)
plt.figure(figsize=(10, 4))
plt.pcolormesh(ts_spec, f_spec, 10 * np.log10(S_spec + 1e-30), shading="gouraud")
plt.ylim(0, 10)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.savefig("step7_spectrogram.png")
plt.close()

H_raw = np.sqrt(N**2 + E**2)
PGV = np.max(H_raw) * 100.0
M = (np.log10(PGV) + np.log10(R) + 1.0) / 0.55

print(f"t_P={t_P:.2f}s, t_S={t_S:.2f}s, t_max={t_max_H:.2f}s")
print(f"R={R:.2f}km (+/-{dR_1s:.2f}km), t_0={t_0:.2f}s")
print(f"PGV={PGV:.4f}cm/s, M={M:.2f}")