import numpy as np


class borgnakke_larssen_model:
    def __init__(self, randomseed: int = 42):
        self.rng = np.random.default_rng(randomseed)

    def collide(self, velocity_i, e_rot_i, velocity_j, e_rot_j, m):
        """
        Perform a collision between two particles using the Borgnakke-Larssen model.

        Args:
            velocity_i: Velocity vector of particle i before collision.
            e_rot_i: Rotational energy of particle i before collision.
            velocity_j: Velocity vector of particle j before collision.
            e_rot_j: Rotational energy of particle j before collision.
            velocity_j: Velocity vector of particle j before collision.
            m: Mass of the particles.
            T: Temperature of the system.

        Returns:
            new_velocity_i: Velocity vector of particle i after collision.
            new_e_rot_i: Rotational energy of particle i after collision.
            new_velocity_j: Velocity vector of particle j after collision.
            new_e_rot_j: Rotational energy of particle j after collision.
        """
        # For transport properties (viscosity, etc.) each binary collision must conserve
        # pair momentum and total energy (translational + internal).
        # We enforce this by working in the center-of-mass (COM) frame.

        inelastic_collision_probability = 1 / 245  # (from Rabitz & Lam 1975)

        # Center-of-mass velocity
        V = 0.5 * (velocity_i + velocity_j)
        g = velocity_i - velocity_j

        if (not np.isfinite(m)) or m <= 0.0:
            return velocity_i, float(e_rot_i), velocity_j, float(e_rot_j)

        E_com = float(m * np.dot(V, V))
        E_rel = float(0.25 * m * np.dot(g, g))
        E_available = E_rel + float(e_rot_i) + float(e_rot_j)

        if (not np.isfinite(E_available)) or E_available < 0.0:
            return velocity_i, float(e_rot_i), velocity_j, float(e_rot_j)

        if self.rng.random() < inelastic_collision_probability:
            # Inelastic collision: redistribute energy
            # For diatomic molecules: 3 translational DOF, 2 rotational DOF per molecule
            translational_fraction = self.rng.beta(2.0, 2.0)
            E_rel_post = E_available * translational_fraction
            E_rot_pool_post = E_available - E_rel_post

            # Split rotational energy between particles
            # For equal molecules with 2 DOF each: Beta(1, 1) = Uniform
            rot_fraction_i = float(self.rng.random())
            new_e_rot_i = E_rot_pool_post * rot_fraction_i
            new_e_rot_j = E_rot_pool_post - new_e_rot_i

            # Sample isotropic relative velocity
            direction = self.rng.normal(size=velocity_i.shape)
            norm = float(np.linalg.norm(direction))
            if norm == 0.0 or (not np.isfinite(norm)):
                direction = np.zeros_like(velocity_i)
                direction[0] = 1.0
                norm = 1.0
            else:
                direction = direction / norm

            g_mag = float(np.sqrt(max(0.0, 4.0 * E_rel_post / m)))
            g_post = direction * g_mag

            new_velocity_i = V + 0.5 * g_post
            new_velocity_j = V - 0.5 * g_post

            return (
                new_velocity_i,
                float(new_e_rot_i),
                new_velocity_j,
                float(new_e_rot_j),
            )
        else:
            # Elastic collision: isotropic hard-sphere deflection
            # Randomize relative velocity direction, preserve speed and COM velocity
            V_com = 0.5 * (velocity_i + velocity_j)
            g_mag = float(np.linalg.norm(g))
            direction = self.rng.normal(size=velocity_i.shape)
            direction /= np.linalg.norm(direction)
            g_post = direction * g_mag
            return (
                V_com + 0.5 * g_post,
                float(e_rot_i),
                V_com - 0.5 * g_post,
                float(e_rot_j),
            )

    def batch_collide(self, velocity_i, e_rot_i, velocity_j, e_rot_j, m):
        """
        Vectorized Borgnakke-Larssen collision for N pairs at once.

        Args:
            velocity_i, velocity_j: (N, 3) numpy arrays
            e_rot_i, e_rot_j: (N,) numpy arrays
            m: scalar mass
        Returns:
            new_v_i, new_e_rot_i, new_v_j, new_e_rot_j
        """
        N = len(velocity_i)
        inelastic_collision_probability = 1/5

        # Center-of-mass frame
        V = 0.5 * (velocity_i + velocity_j)  # (N, 3)
        g = velocity_i - velocity_j  # (N, 3)
        g_speed = np.linalg.norm(g, axis=1)  # (N,)

        E_rel = 0.25 * m * g_speed**2  # (N,)

        # --- Isotropic random scattering direction for all pairs ---
        raw = self.rng.normal(size=(N, 3))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        directions = raw / norms  # (N, 3)

        # --- Default: elastic collision (just rotate g, keep rotational energies) ---
        g_post = directions * g_speed[:, None]
        new_v_i = V + 0.5 * g_post
        new_v_j = V - 0.5 * g_post
        new_e_rot_i = e_rot_i.copy()
        new_e_rot_j = e_rot_j.copy()

        # --- Inelastic collisions: overwrite the selected subset ---
        inelastic = self.rng.random(N) < inelastic_collision_probability
        n_inel = int(np.sum(inelastic))

        if n_inel > 0:
            E_rel_inel = E_rel[inelastic]
            E_available = E_rel_inel + e_rot_i[inelastic] + e_rot_j[inelastic]

            # Energy redistribution
            trans_fraction = self.rng.beta(2.0, 2.0, size=n_inel)
            E_rel_post = E_available * trans_fraction
            E_rot_pool = E_available - E_rel_post

            # Split rotational energy
            rot_fraction = self.rng.random(n_inel)
            new_e_rot_i[inelastic] = E_rot_pool * rot_fraction
            new_e_rot_j[inelastic] = E_rot_pool * (1.0 - rot_fraction)

            # New relative speed from redistributed energy
            g_mag_inel = np.sqrt(np.maximum(0.0, 4.0 * E_rel_post / m))
            g_post_inel = directions[inelastic] * g_mag_inel[:, None]

            new_v_i[inelastic] = V[inelastic] + 0.5 * g_post_inel
            new_v_j[inelastic] = V[inelastic] - 0.5 * g_post_inel

        return new_v_i, new_e_rot_i, new_v_j, new_e_rot_j
