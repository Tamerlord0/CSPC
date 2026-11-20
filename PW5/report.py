import pandas as pd

df = pd.read_csv("results/events.csv")

lines = []
for _, r in df.iterrows():
    lines.append(
        f"Event {r['event']}: P-arrival at {r['t_P_s']} s, S-arrival at {r['t_S_s']} s, "
        f"distance R = {r['R_km']} km, PGV = {r['PGV_cm_s']} cm/s, Magnitude M = {r['M']}."
    )

closest = df.loc[df["R_km"].idxmin(), "event"]
strongest = df.loc[df["M"].idxmax(), "event"]
lines.append(f"\nClosest event: {closest}")
lines.append(f"Strongest event: {strongest}")

with open("results/report.txt", "w") as f:
    f.write("\n".join(lines) + "\n")