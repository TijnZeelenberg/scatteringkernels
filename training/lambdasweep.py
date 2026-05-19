"""Sweep db_lambda ∈ {0.5, 1.0, 5.0} for the DB-penalized MDN."""

from training.core import train_mdn
import paths

LAMBDA_VALUES = [0.5, 1.0, 5.0]

DATAPATH = paths.DATA_DIR / "H2H2_collisions_numba_b1_0_Etr20k_Erot15k_400000_seed42.npy"

TRAIN_KWARGS = dict(
    epochs=100,
    batch_size=2048,
    lr=2.0e-4,
    T_eq=2200.0,
    patience=100,
    showplots=False,
)

if __name__ == "__main__":
    for lam in LAMBDA_VALUES:
        tag = f"db{str(lam).replace('.', '')}"
        outputpath = paths.model_path("mdn", f"mdn_H2_Etr20k_Erot15k_Teq2200_{tag}")
        print(f"\n{'='*60}")
        print(f"db_lambda = {lam}  →  {outputpath}")
        print("=" * 60)
        train_mdn(
            datapath=str(DATAPATH),
            outputpath=str(outputpath),
            db_lambda=lam,
            **TRAIN_KWARGS,
        )
