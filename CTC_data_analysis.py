import pandas as pd
import matplotlib

matplotlib.use("Qt6Agg")
import matplotlib.pyplot as plt


DATA_FILE = "CTC_simulation_results.csv"

df = pd.read_csv(DATA_FILE)

fig, ax = plt.subplots(1, 2, figsize=(10, 6))

ax[0].scatter(0.5 * df["Etr_init_K"], df["Etr1_final_K"], alpha=0.7)
ax[1].scatter(0.5 * df["Etr_init_K"], df["Etr2_final_K"], alpha=0.7)
plt.show()
