import pandas as pd
import numpy as np

INPUT = "data/O2O2_collisions.npy"
OUTPUT = "data/filtered/O2O2_collisions.csv"

data = np.load(INPUT)
data = pd.DataFrame(data, columns=["Etr", "Erot1_in", "Erot2_in", "Etr_out", "Erot1_out", "Erot2_out"])

Etot = data["Etr"] + data["Erot1_in"] + data["Erot2_in"]

# A collision is elastic when both rotational energies are unchanged.
# Use a relative threshold: change < 0.1% of Etot is considered elastic.
delta_rot1 = (data["Erot1_out"] - data["Erot1_in"]).abs()
delta_rot2 = (data["Erot2_out"] - data["Erot2_in"]).abs()
threshold = 0 #0.0001 * Etot

inelastic = (delta_rot1 > threshold) | (delta_rot2 > threshold)

filtered = data[inelastic]
filtered.to_csv(OUTPUT, index=False)

total = len(data)
kept = len(filtered)
print(f"Total collisions:     {total:,}")
print(f"Inelastic collisions: {kept:,} ({100 * kept / total:.1f}%)")
print(f"Elastic (removed):    {total - kept:,} ({100 * (total - kept) / total:.1f}%)")
print(f"Implied Z_rot: {total / kept:.2f}")
print(f"Saved to {OUTPUT}")
