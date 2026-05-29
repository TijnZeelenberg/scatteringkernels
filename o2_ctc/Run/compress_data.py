# run once after generating CSV
import glob
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
matches = glob.glob(os.path.join(script_dir, "O2O2_collisions_uniform_bmax*.csv"))
if len(matches) < 1:
    raise FileNotFoundError(f"Expected 1 matching CSV, found {len(matches)}: {matches}")
file = matches[0]

kB = 1.380649e-23  # Boltzmann constant in J/K

data = np.loadtxt(file, delimiter=",", skiprows=1)

# convert to Kelvin
data = data / kB

# Columns: Etr, Erot1_in, Erot2_in, Etr_out, Erot1_out, Erot2_out
etr_in = data[:, 0]
etr_out = data[:, 3]
delta_over_etr = (etr_out - etr_in) / etr_in
print(f"mean       (Etr_out - Etr_in) / Etr_in : {delta_over_etr.mean():+.4e}")
print(f"mean |(Etr_out - Etr_in) / Etr_in|     : {np.abs(delta_over_etr).mean():.4e}")

np.save(file.replace(".csv", ".npy"), data)

# Remove the original CSV file to save space
os.remove(file)
