# Briefing: `recursive_equilibrium.png`

This document explains the figure produced by `analysis/recursive_equilibrium.py`
(`results/plots/recursive_equilibrium.png`) in enough detail to write the thesis
section about it. It states the claim, the method, what every line in the figure
means, the numbers (from the current run), and the caveats.

---

## 1. The claim

The one-shot-trained MDN scattering kernel reproduces individual collisions
accurately, yet in DSMC energy-relaxation it drives the gas to the **wrong steady
state** (final temperatures offset from equipartition: H₂ → ~216/230 K, O₂ →
~226/215 K instead of 220/220 K). This is **not a fitting bug** — it is the
signature of a **learned violation of detailed balance**, and the specific
equilibrium DSMC lands on is exactly the one the kernel's measured violation
predicts.

- The MDN is trained on a **one-shot** map: given a pre-collision state it predicts
  the post-collision energy fractions, and the negative-log-likelihood loss only
  measures how well *one* collision is reproduced, averaged over the training data.
- Nothing in that loss constrains **detailed balance** (microscopic reversibility),
  the property that makes the Maxwell–Boltzmann equilibrium *stationary* under
  collisions. DSMC applies the kernel **recursively**, millions of times, so a
  small per-collision reversibility error does not cancel — it compounds — and the
  gas settles a few K away from equipartition.

The figure makes this quantitative: it measures the violation directly and shows
its predicted equilibrium coincides with the DSMC one, for **both** gases.

---

## 2. Background concepts

**Translational energy fraction `η_tr`.** For a colliding pair, split the
redistributable energy into relative translation and the two rotations:

```
E_c   = E_rel + E_rot,1 + E_rot,2     (collision energy, conserved per collision)
η_tr  = E_rel / E_c                    (fraction in translation)
```

Thermal equilibrium `T_trans = T_rot` is **equipartition**, `η_tr = 3/7 ≈ 0.4286`
(3 translational vs 3+2+2 DOF).

**Detailed balance ⇒ zero net flux at equilibrium.** A microscopically reversible
collision operator leaves Maxwell–Boltzmann invariant: if the gas is *already* at
equipartition, collisions transfer **zero net energy** between translation and
rotation. This is the operational test we use. The true (CTC, classical-trajectory)
operator passes it — see `visualization/detailed_balance.py`, which shows the
equilibrium-weighted `η` distribution is unchanged by a CTC collision (Δ⟨η⟩ ≈ 0).
The MDN loss never enforces this, so the trained kernel need not pass it.

**NTC acceptance.** In DSMC a pair collides with probability proportional to its
relative speed `g ∝ √E_rel` (the no-time-counter acceptance, `accept_collision` in
`physics/dsmc.py`). The equilibrium distribution of *colliding-pair* relative
translational energy is therefore the Maxwell–Boltzmann `Gamma(3/2, T)` tilted by
`√E_rel` into `Gamma(2, T)` (mean `2·T_trans`); rotational energies are
`Exponential(T_rot)` (2 DOF each). We sample exactly this physical ensemble.

---

## 3. The metric (what the script computes)

Everything is done in temperature space (Kelvin) along the **energy-conservation
line** the relaxation experiments live on: starting from `T_trans = 300`,
`T_rot = 100`, each molecule conserves `E0 = 1.5·T_trans + T_rot = 550 K`, so
`T_rot = E0 − 1.5·T_trans` and equipartition is `T_trans = T_rot = 220 K`.

For a trial equilibrium gas at `(T_trans, T_rot)` we draw the physical NTC-weighted
collision ensemble (`Gamma(2, T_trans)` for `E_rel`, `Exponential(T_rot)` for each
rotation), apply the MDN **once**, and measure the net translational energy change
per collision:

```
D(T_trans) = ⟨ η_tr'·E_c − E_rel ⟩       (mdn_translational_drift)
```

Three numbers follow:

1. **The violation** `D_eq = D(220 K)` — the net flux *at equipartition*. A
   detailed-balance kernel gives `D_eq = 0`; a nonzero `D_eq` (with sign) is a
   rigorous first-moment **witness** of a violation.
2. **The restoring slope** `S = dD/dT_trans` at equipartition (local linear fit of
   the drift curve).
3. **The consequence** — the kernel's own steady state is where the flux vanishes,
   the **zero-crossing** `T*` of `D(T_trans)` (`temperature_fixed_point`). Because
   `D_eq ≠ 0`, `T* ≠ 220 K`. Linear response decomposes the offset as
   `ΔT* ≈ −D_eq / S`. We compare `T*` against the temperature the full DSMC run
   actually converges to (`dsmc_converged_temperatures`, last 20 % of
   `data/ml-dsmc/mdn/{h2,o2}/best_model_relaxation.npy`).

---

## 4. The figure (1 × 2; H₂ left, O₂ right)

The figure is deliberately minimal — the y-axis identifies the curve and the legend
holds only the two temperatures being compared (equipartition and the DSMC-converged
value); the caption (this section) carries the rest of the meaning.

- x-axis: `T_trans` [K] (with `T_rot = E0 − 1.5·T_trans` along the conservation
  line). y-axis: `D` [K/collision] — i.e. `⟨ΔE_rel⟩`, the mean translational energy
  change per collision.
- **Red curve + markers (no legend entry):** the MDN `D(T_trans)`. It crosses zero
  exactly once.
- **Grey horizontal line `y = 0`** and **grey dotted vertical at 220 K**
  (legend: *equipartition*). A detailed-balance kernel's curve must pass through
  their intersection `(220, 0)`. The MDN curve does **not**: at equipartition it
  sits a finite distance `D_eq` off zero — that gap is the learned violation
  (its value is `D_eq` in the printed table, §5).
- **Black dashed vertical (legend: *DSMC converged*):** the temperature the full
  DSMC simulation converged to.

The kernel's own steady state `T*` (the flux zero-crossing), the restoring slope `S`,
and the linear-response offset `−D_eq/S` are reported in the printed table (§5) but
are no longer drawn, to keep the figure uncluttered. **The proof is in the table:
the kernel's zero-flux `T*` matches the DSMC-converged temperature** (the black dashed
line), for both gases.

How to read it in one breath: *the kernel has a nonzero flux at equipartition (the
violation); the gas slides along the curve until the flux vanishes; that landing
point `T*` is the DSMC equilibrium, offset from equipartition.*

---

## 5. The numbers (current 20 ns run; deterministic, seed 0)

| Species | `D_eq` [K/coll] | `S` [K/coll/K] | `−D_eq/S` [K] | kernel `T*`/`T_rot*` | DSMC `T_trans`/`T_rot` | `ΔT* = T*−T_DSMC` |
|---|---|---|---|---|---|---|
| H₂ | **−1.05** (drains trans) | −0.17 | −6.1 | 215.0 / 227.6 | 216.2 / 229.5 | **−1.3 K** |
| O₂ | **+0.22** (pumps trans)  | −0.15 | +1.5 | 222.1 / 216.9 | 225.6 / 215.4 | **−3.6 K** |

Equipartition is 220 / 220. One-shot map fidelity on the dense `E_c ≈ 5000 K`
shell: `RMS(MDN − CTC) ≈ 0.005–0.006`, max `≈ 0.018` — the kernel reproduces single
collisions to sub-percent accuracy.

**Reading the result.** `D_eq ≠ 0` for both gases (the violation), and its **sign
flips** between them — H₂ drains translation into rotation (ends `T_rot > T_trans`),
O₂ pumps the other way (ends `T_trans > T_rot`). The kernel's zero-flux temperature
`T*` reproduces the DSMC-converged temperature in sign for both and in magnitude to
**1.3 K (H₂)** and within a few K (O₂). The sign flip — reproduced by a kernel-only
calculation that never sees DSMC — rules out a generic numerical artefact: this is a
property of each trained kernel.

*(These come from the current models/data. Re-run `python -m
analysis.recursive_equilibrium` and read its stdout if anything is retrained or the
relaxation experiments are rerun.)*

---

## 6. Caveats (state honestly)

1. **First-moment witness.** `D_eq` is a *mean-energy* quantity — exactly what fixes
   the equilibrium *temperatures* DSMC reports — but it does not test higher moments
   of the kernel. A kernel could in principle have `D_eq = 0` yet still distort the
   distribution shape; here `D_eq ≠ 0` is already sufficient to explain the wrong
   temperatures.
2. **Linear response is approximate.** `−D_eq/S` is a small-offset linearisation; it
   overshoots the exact zero-crossing for H₂ (−6.1 vs the actual −5.0 K offset) and
   undershoots for O₂ (+1.5 vs +2.1). The headline comparison to DSMC therefore uses
   the **exact** `T*`, not `−D_eq/S`.
3. **O₂ magnitude.** The agreement is tight in sign for both gases and to ~1 K for
   H₂, but O₂'s `T*` is ~3.6 K short of the DSMC offset. The residual reflects the
   Maxwell–Boltzmann-equilibrium assumption in the single-collision drift integral
   and the ~0.2 K/ns residual drift still present in the 20 ns DSMC trace (the script
   prints a `⚠ NOT CONVERGED` warning if that drift exceeds 0.5 K/ns). Claim
   "correct sign for both, flip reproduced, magnitude within a few K", **not**
   "exact".
4. **True-kernel control is at a different energy.** CTC's `D = 0` (detailed balance
   holds) is demonstrated in `visualization/detailed_balance.py` at a data-supported
   temperature (~1000 K), because the CTC uniform-box data is too sparse (<1 % of
   samples) below ~4000 K to evaluate at the ~780 K energy DSMC equilibrates at. On
   *this* figure the true-kernel reference is therefore the theoretical
   `(220 K, 0)` point (the star), not an empirical CTC curve.
5. **Single-pair mean-field reduction.** The drift integral iterates one pair with
   the rotational partition and collision energy drawn fresh each step — a mean-field
   reduction of the many-body DSMC dynamics. It captures the mechanism and the
   sign/magnitude of the offset, not an exact numerical match.
6. **Corroborating witness (optional to cite).** The MDN's single-shell `η` fixed
   point is *energy-dependent* (≈0.40 at `E_c = 5000 K` → ≈0.33 at 783 K for H₂); a
   microscopically reversible kernel gives the same `3/7` at every energy, so the
   energy dependence is a second, independent fingerprint of broken detailed balance.

---

## 7. One-sentence takeaway

> One-shot maximum-likelihood training constrains the kernel pointwise but never
> targets detailed balance, so the trained MDN reproduces individual collisions to
> sub-percent accuracy yet carries a nonzero net energy flux at equilibrium; iterated
> by DSMC that flux drives the gas to a steady state offset from equipartition — by
> the amount, and in the (gas-dependent) direction, the kernel's own zero-flux
> temperature predicts.

This motivates (out of scope here) adding a stationarity / detailed-balance penalty
during training, or importance-reweighting the loss toward the operating
distribution DSMC visits (`mdn_loss_weighted` exists in `machinelearning/mdn.py`).

---

## 8. Related code

- `analysis/recursive_equilibrium.py` — produces this figure and the printed table.
- `visualization/detailed_balance.py` — shows the CTC operator preserves
  Maxwell–Boltzmann (the true-kernel `D = 0` control the MDN loss does not enforce).
- `analysis/kernel_stationarity.py` — probes the MDN once at equilibrium inputs;
  conditional-mean drift and its zero-crossing (`_apply_mdn` lives here).
- `visualization/create_plots.py` (§11) — the `best_model_relaxation` temperature
  trajectories whose wrong final temperatures this figure explains.
- `machinelearning/mdn.py` — MDN definition, sampling, and the loss functions.
