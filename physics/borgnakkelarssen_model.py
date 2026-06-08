import numpy as np


class borgnakke_larssen_model:
    """Borgnakke-Larsen rotational relaxation, matching SPARTA's serial
    per-molecule Larsen-Borgnakke scheme (``collide_vss.cpp``,
    ``EEXCHANGE_NonReactingEDisposal``).

    Per collision, *each* of the two molecules independently relaxes its
    rotational energy with probability ``1/zrot``. When a molecule relaxes it
    draws a new rotational energy equal to ``(1 - U**(1/(2.5 - omega)))`` times
    the running collision energy ``E_c`` (relative translational energy plus
    that molecule's rotational energy); the translational pool is then updated
    and carried over to the second molecule. After both molecules are handled,
    the post-collision relative velocity is scattered isotropically (VHS/VSS
    with ``alpha = 1``).

    ``omega`` is the VHS viscosity-temperature exponent. ``omega = 0.5`` reduces
    to the hard-sphere model, which is what the H2/O2 ``*.vhs`` parameter files
    use (so it is the default).
    """

    def __init__(self, randomseed: int = 42, omega: float = 0.5):
        self.rng = np.random.default_rng(randomseed)
        self.omega = omega

    def _isotropic_directions(self, shape):
        """Unit vectors drawn uniformly on the sphere, with a zero-norm guard.

        ``shape`` is the full output shape, e.g. ``(3,)`` for a single pair or
        ``(N, 3)`` for a batch; the last axis is normalized.
        """
        raw = self.rng.normal(size=shape)
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return raw / norms

    def collide(self, velocity_i, e_rot_i, velocity_j, e_rot_j, m, zrot: float = 1.0):
        """Perform one Borgnakke-Larsen collision (SPARTA serial scheme).

        Args:
            velocity_i: Velocity vector of particle i before collision.
            e_rot_i: Rotational energy of particle i before collision.
            velocity_j: Velocity vector of particle j before collision.
            e_rot_j: Rotational energy of particle j before collision.
            m: Mass of the particles.
            zrot: Rotational collision number; the inelastic probability is
                ``1/zrot`` tested independently for each molecule.

        Returns:
            new_velocity_i, new_e_rot_i, new_velocity_j, new_e_rot_j
        """
        # Work in the centre-of-mass frame: V is conserved (pair momentum) and
        # the redistribution conserves relative translational + rotational
        # energy.
        V = 0.5 * (velocity_i + velocity_j)
        g = velocity_i - velocity_j

        if (not np.isfinite(m)) or m <= 0.0:
            return velocity_i, float(e_rot_i), velocity_j, float(e_rot_j)

        # Relative translational energy = 1/2 * mu * g^2 with reduced mass m/2.
        E_trans = float(0.25 * m * np.dot(g, g))
        if (not np.isfinite(E_trans)) or E_trans < 0.0:
            return velocity_i, float(e_rot_i), velocity_j, float(e_rot_j)

        phi = 1.0 / zrot
        exponent = 1.0 / (2.5 - self.omega)

        E_dispose = E_trans
        new_e_rot_i = float(e_rot_i)
        new_e_rot_j = float(e_rot_j)

        # Molecule i relaxes its rotation against the running translational pool.
        if self.rng.random() < phi:
            E_c = E_dispose + new_e_rot_i
            fraction_rot = 1.0 - self.rng.random() ** exponent
            new_e_rot_i = fraction_rot * E_c
            E_dispose = E_c - new_e_rot_i

        # Molecule j relaxes against the pool already updated by molecule i.
        if self.rng.random() < phi:
            E_c = E_dispose + new_e_rot_j
            fraction_rot = 1.0 - self.rng.random() ** exponent
            new_e_rot_j = fraction_rot * E_c
            E_dispose = E_c - new_e_rot_j

        # Isotropic scatter using the post-relaxation relative translational
        # energy (speed is preserved when neither molecule relaxed -> elastic).
        direction = self._isotropic_directions(np.shape(velocity_i))
        g_mag = float(np.sqrt(max(0.0, 4.0 * E_dispose / m)))
        g_post = direction * g_mag

        return (
            V + 0.5 * g_post,
            float(new_e_rot_i),
            V - 0.5 * g_post,
            float(new_e_rot_j),
        )

    def batch_collide(
        self, velocity_i, e_rot_i, velocity_j, e_rot_j, m, zrot: float = 1.0
    ):
        """Vectorized SPARTA serial Borgnakke-Larsen collision for N pairs.

        Args:
            velocity_i, velocity_j: (N, 3) arrays of pre-collision velocities.
            e_rot_i, e_rot_j: (N,) arrays of pre-collision rotational energies.
            m: scalar particle mass.
            zrot: rotational collision number; inelastic probability ``1/zrot``
                is tested independently for each molecule of each pair.

        Returns:
            new_v_i, new_e_rot_i, new_v_j, new_e_rot_j
        """
        N = len(velocity_i)

        # Centre-of-mass frame.
        V = 0.5 * (velocity_i + velocity_j)  # (N, 3)
        g = velocity_i - velocity_j  # (N, 3)
        g_speed = np.linalg.norm(g, axis=1)  # (N,)

        # Relative translational energy = 1/2 * mu * g^2 with reduced mass m/2.
        E_trans = 0.25 * m * g_speed**2  # (N,)

        phi = 1.0 / zrot
        exponent = 1.0 / (2.5 - self.omega)

        E_dispose = E_trans.copy()
        new_e_rot_i = e_rot_i.copy()
        new_e_rot_j = e_rot_j.copy()

        # Molecule i: own 1/zrot test, relax against the translational pool.
        relax_i = self.rng.random(N) < phi
        E_c_i = E_dispose + new_e_rot_i
        e_i_relaxed = (1.0 - self.rng.random(N) ** exponent) * E_c_i
        new_e_rot_i = np.where(relax_i, e_i_relaxed, new_e_rot_i)
        E_dispose = np.where(relax_i, E_c_i - e_i_relaxed, E_dispose)

        # Molecule j: own 1/zrot test, sees E_dispose already updated by i.
        relax_j = self.rng.random(N) < phi
        E_c_j = E_dispose + new_e_rot_j
        e_j_relaxed = (1.0 - self.rng.random(N) ** exponent) * E_c_j
        new_e_rot_j = np.where(relax_j, e_j_relaxed, new_e_rot_j)
        E_dispose = np.where(relax_j, E_c_j - e_j_relaxed, E_dispose)

        # Isotropic scatter using the post-relaxation relative translational
        # energy (speed preserved when neither molecule relaxed -> elastic).
        directions = self._isotropic_directions((N, 3))
        g_mag = np.sqrt(np.maximum(0.0, 4.0 * E_dispose / m))  # (N,)
        g_post = directions * g_mag[:, None]

        new_v_i = V + 0.5 * g_post
        new_v_j = V - 0.5 * g_post

        return new_v_i, new_e_rot_i, new_v_j, new_e_rot_j
