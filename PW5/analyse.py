import os
import numpy as np
import pandas as pd

def bandpass(x, fs, a, b):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), d=1.0 / fs)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0
    return np.fft.ifft(X).real

def envelope(x, fs, win_s):
    w = max(1, int(round(win_s * fs)))
    return np.convolve(np.abs(x), np.ones(w) / w, mode="same")

def analyse(csv_file):
    d = pd.read_csv(csv_file)
    t = d["time_s"].to_numpy()
    Z = d["Z"].to_numpy()
    N = d["N"].to_numpy()
    E = d["E"].to_numpy()
    fs = 1.0 / (t[1] - t[0])

    Z_f = bandpass(Z, fs, 0.5, 10.0)
    N_f = bandpass(N, fs, 0.5, 10.0)
    E_f = bandpass(E, fs, 0.5, 10.0)

    H_f = np.sqrt(N_f**2 + E_f**2)
    env_Z = envelope(Z_f, fs, 0.5)
    env_H = envelope(H_f, fs, 0.5)

    t_P = t[np.argmax(env_Z > 0.05 * np.max(env_Z))]
    mask_S = t > (t_P + 2.0)
    t_S = t[mask_S][np.argmax(env_H[mask_S] > 0.10 * np.max(env_H))]

    v_P, v_S = 6.2, 3.6
    R = (t_S - t_P) / (1.0 / v_S - 1.0 / v_P)

    PGV = np.max(np.sqrt(N**2 + E**2)) * 100.0
    M = (np.log10(PGV) + np.log10(R) + 1.0) / 0.55

    return {
        "event": os.path.basename(csv_file),
        "t_P_s": round(float(t_P), 3),
        "t_S_s": round(float(t_S), 3),
        "R_km": round(float(R), 3),
        "PGV_cm_s": round(float(PGV), 5),
        "M": round(float(M), 3)
    }

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    files = ["data/seismogram_A.csv", "data/seismogram_B.csv"]
    rows = [analyse(f) for f in files]
    pd.DataFrame(rows).to_csv("results/events.csv", index=False)