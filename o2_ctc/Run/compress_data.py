# run once after generating CSV
import numpy as np
import os

file = "O2O2_collisions_uniform.csv"

kB = 1.380649e-23  # Boltzmann constant in J/K

data = np.loadtxt(file, delimiter=",", skiprows=1)

# convert to Kelvin
data = data / kB
np.save(file.replace(".csv", ".npy"), data)

# Remove the original CSV file to save space
os.remove(file)
