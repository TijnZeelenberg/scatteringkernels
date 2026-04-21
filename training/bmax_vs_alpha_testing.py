from training.trainer import train_mdn

bmax = [1.0, 1.2, 1.5]
alpha = [0.5, 1.0, 1.5]

datasets = [
    "data/H2H2_collisions_numba_b1_0.npy",
    "data/H2H2_collisions_numba_b1_2.npy",
    "data/H2H2_collisions_numba_b1_5.npy",
]

for a in alpha:
    for dataset, b in zip(datasets, bmax):
        outputpath = f"results/models/H2H2_mdn_b{b}_alpha{a}".replace(".", "_") + ".pth"
        train_mdn(datapath=dataset, outputpath=outputpath, wf=a)
