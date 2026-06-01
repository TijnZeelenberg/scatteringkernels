"""Per-collision diagnostics for DSMC runs.

Attach a `CollisionLogger` to `DSMC_Simulation.run_simulation` to dump:

  * Cheap aggregates every step — n_collisions, pre-collision E_rel / E_rot /
    E_total moments, mean and std of dE_trans, energy-conservation residual,
    and (optionally) the fraction of inputs outside the training caps.
  * Full per-collision arrays at selected snapshot steps — particle indices,
    pre and post velocities, and pre and post rotational energies.

All energies are stored in Joules (matching `DSMC_Simulation`'s internal units);
convert to Kelvin via `/ 1.380649e-23` at analysis time.

Snapshots have variable length per step, so they are concatenated end-to-end
and indexed via a `snapshot_offsets[K+1]` array — slice snapshot k with
``arr[snapshot_offsets[k] : snapshot_offsets[k+1]]``.

Output is written via `np.savez_compressed` to a `.npz` archive.
"""

from pathlib import Path

import numpy as np

_KB = 1.380649e-23


class CollisionLogger:
    def __init__(
        self,
        output_path: str | Path,
        snapshot_every: int | None = None,
        training_caps_K: dict | None = None,
    ):
        """
        Args:
            output_path: where to write the `.npz` archive on `finalize()`.
            snapshot_every: if set, also dump full per-collision arrays every
                N steps. `None` disables full snapshots (aggregates only).
            training_caps_K: optional `{"E_trans_max_K": float,
                "E_rot_max_K": float}` (per-particle caps in Kelvin/kB units)
                used to compute the per-step out-of-distribution fraction.
        """
        self.output_path = Path(output_path)
        self.snapshot_every = snapshot_every
        self.training_caps_K = training_caps_K

        self._steps: list[int] = []
        self._n_collisions: list[int] = []
        self._E_rel_pre_mean: list[float] = []
        self._E_rel_pre_std: list[float] = []
        self._E_rot_pair_pre_mean: list[float] = []
        self._E_rot_pair_pre_std: list[float] = []
        self._E_total_pre_mean: list[float] = []
        self._E_total_pre_std: list[float] = []
        self._delta_E_trans_mean: list[float] = []
        self._delta_E_trans_std: list[float] = []
        self._energy_residual_max: list[float] = []
        self._frac_ood: list[float] = []

        self._snapshots: list[dict] = []

    def log_step(
        self,
        step: int,
        mass: float,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
        v_i: np.ndarray,
        v_j: np.ndarray,
        e_rot_i: np.ndarray,
        e_rot_j: np.ndarray,
        new_v_i: np.ndarray,
        new_v_j: np.ndarray,
        new_e_rot_i: np.ndarray,
        new_e_rot_j: np.ndarray,
    ) -> None:
        n = int(idx_i.shape[0])
        self._steps.append(int(step))
        self._n_collisions.append(n)

        if n == 0:
            self._E_rel_pre_mean.append(np.nan)
            self._E_rel_pre_std.append(np.nan)
            self._E_rot_pair_pre_mean.append(np.nan)
            self._E_rot_pair_pre_std.append(np.nan)
            self._E_total_pre_mean.append(np.nan)
            self._E_total_pre_std.append(np.nan)
            self._delta_E_trans_mean.append(np.nan)
            self._delta_E_trans_std.append(np.nan)
            self._energy_residual_max.append(np.nan)
            self._frac_ood.append(np.nan)
            return

        mu = 0.5 * mass  # reduced mass for identical particles
        v_rel_pre = v_i - v_j
        E_rel_pre = 0.5 * mu * np.sum(v_rel_pre * v_rel_pre, axis=1)
        v_rel_post = new_v_i - new_v_j
        E_rel_post = 0.5 * mu * np.sum(v_rel_post * v_rel_post, axis=1)

        E_rot_pair_pre = e_rot_i + e_rot_j
        E_rot_pair_post = new_e_rot_i + new_e_rot_j
        E_total_pre = E_rel_pre + E_rot_pair_pre
        E_total_post = E_rel_post + E_rot_pair_post

        delta_E_trans = E_rel_post - E_rel_pre
        residual = E_total_post - E_total_pre

        self._E_rel_pre_mean.append(float(np.mean(E_rel_pre)))
        self._E_rel_pre_std.append(float(np.std(E_rel_pre)))
        self._E_rot_pair_pre_mean.append(float(np.mean(E_rot_pair_pre)))
        self._E_rot_pair_pre_std.append(float(np.std(E_rot_pair_pre)))
        self._E_total_pre_mean.append(float(np.mean(E_total_pre)))
        self._E_total_pre_std.append(float(np.std(E_total_pre)))
        self._delta_E_trans_mean.append(float(np.mean(delta_E_trans)))
        self._delta_E_trans_std.append(float(np.std(delta_E_trans)))
        self._energy_residual_max.append(float(np.max(np.abs(residual))))

        if self.training_caps_K is not None:
            E_trans_max = self.training_caps_K.get("E_trans_max_K", np.inf) * _KB
            E_rot_max = self.training_caps_K.get("E_rot_max_K", np.inf) * _KB
            ood = (
                (E_rel_pre > E_trans_max)
                | (e_rot_i > E_rot_max)
                | (e_rot_j > E_rot_max)
            )
            self._frac_ood.append(float(np.mean(ood)))
        else:
            self._frac_ood.append(np.nan)

        if self.snapshot_every and (step % self.snapshot_every == 0):
            self._snapshots.append(
                {
                    "step": int(step),
                    "idx_i": np.asarray(idx_i, dtype=np.int64).copy(),
                    "idx_j": np.asarray(idx_j, dtype=np.int64).copy(),
                    "v_i_pre": np.asarray(v_i, dtype=np.float32).copy(),
                    "v_j_pre": np.asarray(v_j, dtype=np.float32).copy(),
                    "v_i_post": np.asarray(new_v_i, dtype=np.float32).copy(),
                    "v_j_post": np.asarray(new_v_j, dtype=np.float32).copy(),
                    "e_rot_i_pre": np.asarray(e_rot_i, dtype=np.float32).copy(),
                    "e_rot_j_pre": np.asarray(e_rot_j, dtype=np.float32).copy(),
                    "e_rot_i_post": np.asarray(new_e_rot_i, dtype=np.float32).copy(),
                    "e_rot_j_post": np.asarray(new_e_rot_j, dtype=np.float32).copy(),
                }
            )

    def finalize(self) -> Path:
        out: dict[str, np.ndarray] = {
            "step": np.array(self._steps, dtype=np.int64),
            "n_collisions": np.array(self._n_collisions, dtype=np.int64),
            "E_rel_pre_mean": np.array(self._E_rel_pre_mean),
            "E_rel_pre_std": np.array(self._E_rel_pre_std),
            "E_rot_pair_pre_mean": np.array(self._E_rot_pair_pre_mean),
            "E_rot_pair_pre_std": np.array(self._E_rot_pair_pre_std),
            "E_total_pre_mean": np.array(self._E_total_pre_mean),
            "E_total_pre_std": np.array(self._E_total_pre_std),
            "delta_E_trans_mean": np.array(self._delta_E_trans_mean),
            "delta_E_trans_std": np.array(self._delta_E_trans_std),
            "energy_residual_max": np.array(self._energy_residual_max),
            "frac_ood": np.array(self._frac_ood),
        }

        if self._snapshots:
            lengths = [int(s["idx_i"].shape[0]) for s in self._snapshots]
            offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
            out["snapshot_steps"] = np.array(
                [s["step"] for s in self._snapshots], dtype=np.int64
            )
            out["snapshot_offsets"] = offsets
            for key in (
                "idx_i",
                "idx_j",
                "v_i_pre",
                "v_j_pre",
                "v_i_post",
                "v_j_post",
                "e_rot_i_pre",
                "e_rot_j_pre",
                "e_rot_i_post",
                "e_rot_j_post",
            ):
                arrays = [s[key] for s in self._snapshots]
                out[f"snapshot_{key}"] = np.concatenate(arrays, axis=0)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_path, **out)
        return self.output_path
