"""Conservation invariants shared by every collision model.

The DSMC engine dispatches on duck typing (`physics/dsmc.py`: `hasattr(model,
"batch_collide")`), so any object can be plugged in as a collision model. The
only thing the engine relies on is that a collision conserves pair momentum and
total redistributable energy. These tests pin that contract down for all three
models, and pin the scalar and vectorized paths of the Borgnakke-Larssen
baseline to the same redistribution law.
"""

import numpy as np
import pytest
import torch
from scipy import stats

from machinelearning.beta_mdn import BetaMixtureDensityNetwork
from machinelearning.mdn import MixtureDensityNetwork
from physics.borgnakkelarssen_model import borgnakke_larssen_model

N_PAIRS = 2000
MASS = 3.34e-27 * 2  # H2 pair mass parameter [kg]
SEED = 7

# energy bookkeeping
# Models take `m` such that E_rel = 0.25 * m * |v_i - v_j|^2, and return
# post-collision velocities around the same centre of mass.


def total_energy(v_i, e_rot_i, v_j, e_rot_j, m):
    g = v_i - v_j
    e_rel = 0.25 * m * np.sum(np.atleast_2d(g) ** 2, axis=-1)
    return e_rel + np.asarray(e_rot_i) + np.asarray(e_rot_j)


def random_pairs(n=N_PAIRS, seed=SEED):
    rng = np.random.default_rng(seed)
    thermal_speed = np.sqrt(1.380649e-23 * 300.0 / MASS)
    v_i = rng.normal(scale=thermal_speed, size=(n, 3))
    v_j = rng.normal(scale=thermal_speed, size=(n, 3))
    e_rot_i = rng.exponential(scale=1.380649e-23 * 300.0, size=n)
    e_rot_j = rng.exponential(scale=1.380649e-23 * 300.0, size=n)
    return v_i, e_rot_i, v_j, e_rot_j


# model fixtures
# The MDN models normalize their inputs using statistics computed in
# create_dataloaders. Conservation holds for any weights, trained or not, so the
# tests prime the normalization on synthetic data rather than requiring a
# checkpoint.


def _prime_normalization(model):
    rng = np.random.default_rng(SEED)
    n = 512
    e_total = rng.uniform(1e-21, 1e-19, size=n) / 1.380649e-23
    X = torch.tensor(
        np.stack([e_total, rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.95, n)], 1),
        dtype=torch.float32,
    )
    y = torch.tensor(
        np.stack([rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.95, n)], 1),
        dtype=torch.float32,
    )
    model.create_dataloaders(
        X, y, batch_size=64, shuffle=True, trainval_split=0.7, random_seed=SEED
    )
    return model


@pytest.fixture(
    params=["borgnakke_larssen", "mdn", "beta_mdn"],
    ids=["BL", "MDN", "BetaMDN"],
)
def collision_model(request):
    if request.param == "borgnakke_larssen":
        return borgnakke_larssen_model(randomseed=SEED)
    cls = MixtureDensityNetwork if request.param == "mdn" else BetaMixtureDensityNetwork
    model = cls(
        input_dim=3, output_dim=2, num_mixtures=5, hidden_dim=8, randomseed=SEED
    )
    return _prime_normalization(model)


# tests


@pytest.mark.parametrize("zrot", [1.0, 5.0])
def test_batch_collide_conserves_energy(collision_model, zrot):
    v_i, e_rot_i, v_j, e_rot_j = random_pairs()
    before = total_energy(v_i, e_rot_i, v_j, e_rot_j, MASS)

    nv_i, ne_rot_i, nv_j, ne_rot_j = collision_model.batch_collide(
        v_i, e_rot_i, v_j, e_rot_j, MASS, zrot=zrot
    )
    after = total_energy(nv_i, ne_rot_i, nv_j, ne_rot_j, MASS)

    np.testing.assert_allclose(after, before, rtol=1e-5)


@pytest.mark.parametrize("zrot", [1.0, 5.0])
def test_batch_collide_conserves_momentum(collision_model, zrot):
    v_i, e_rot_i, v_j, e_rot_j = random_pairs()

    nv_i, _, nv_j, _ = collision_model.batch_collide(
        v_i, e_rot_i, v_j, e_rot_j, MASS, zrot=zrot
    )

    np.testing.assert_allclose(nv_i + nv_j, v_i + v_j, rtol=1e-5, atol=1e-12)


def test_batch_collide_keeps_rotational_energy_non_negative(collision_model):
    v_i, e_rot_i, v_j, e_rot_j = random_pairs()

    _, ne_rot_i, _, ne_rot_j = collision_model.batch_collide(
        v_i, e_rot_i, v_j, e_rot_j, MASS, zrot=1.0
    )

    assert np.all(ne_rot_i >= 0.0)
    assert np.all(ne_rot_j >= 0.0)


def test_bl_scalar_path_conserves_energy():
    model = borgnakke_larssen_model(randomseed=SEED)
    v_i, e_rot_i, v_j, e_rot_j = random_pairs(n=200)

    for k in range(len(v_i)):
        before = total_energy(v_i[k], e_rot_i[k], v_j[k], e_rot_j[k], MASS)
        out = model.collide(v_i[k], e_rot_i[k], v_j[k], e_rot_j[k], MASS, zrot=1.0)
        after = total_energy(out[0], out[1], out[2], out[3], MASS)
        np.testing.assert_allclose(after, before, rtol=1e-6)
        np.testing.assert_allclose(
            out[0] + out[2], v_i[k] + v_j[k], rtol=1e-6, atol=1e-12
        )


def test_bl_scalar_and_batch_paths_agree():
    """The two BL code paths must implement the same redistribution law.

    They silently diverged once (Beta(1.5, 2) vs Beta(2, 2)) because nothing
    compared them. Both now read `TRANS_FRACTION_BETA`; this test is what keeps
    them that way.
    """
    v_i, e_rot_i, v_j, e_rot_j = random_pairs(n=4000)
    e_total = total_energy(v_i, e_rot_i, v_j, e_rot_j, MASS)

    batch_model = borgnakke_larssen_model(randomseed=SEED)
    nv_i, _, nv_j, _ = batch_model.batch_collide(
        v_i, e_rot_i, v_j, e_rot_j, MASS, zrot=1.0
    )
    g = nv_i - nv_j
    eta_batch = (0.25 * MASS * np.sum(g**2, axis=1)) / e_total

    scalar_model = borgnakke_larssen_model(randomseed=SEED + 1)
    eta_scalar = np.empty(len(v_i))
    for k in range(len(v_i)):
        out = scalar_model.collide(
            v_i[k], e_rot_i[k], v_j[k], e_rot_j[k], MASS, zrot=1.0
        )
        g_k = out[0] - out[2]
        eta_scalar[k] = (0.25 * MASS * np.dot(g_k, g_k)) / e_total[k]

    assert stats.ks_2samp(eta_batch, eta_scalar).pvalue > 0.01
