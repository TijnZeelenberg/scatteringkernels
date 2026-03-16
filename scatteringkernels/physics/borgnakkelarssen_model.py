import numpy as np

class borgnakke_larssen_model:

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def postsample(self, velocity_i, e_rot_i, velocity_j, e_rot_j, m, T):
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
        # NOTE:
        # For transport properties (viscosity, etc.) each binary collision must conserve
        # pair momentum and total energy (translational + internal).
        # We enforce this by working in the center-of-mass (COM) frame.

        inelastic_collision_probability = 1 / 245  # Rabitz & Lam (1975)

        # Center-of-mass velocity (equal masses assumed)
        V = 0.5 * (velocity_i + velocity_j)
        g = velocity_i - velocity_j
        if (not np.isfinite(m)) or m <= 0.0:
            return velocity_i, float(e_rot_i), velocity_j, float(e_rot_j)

        # Energy decomposition:
        # E_total = E_com + E_rel + E_rot_i + E_rot_j
        E_com = float(m * np.dot(V, V))
        E_rel = float(0.25 * m * np.dot(g, g))
        E_available = E_rel + float(e_rot_i) + float(e_rot_j)
        if (not np.isfinite(E_available)) or E_available < 0.0:
            return velocity_i, float(e_rot_i), velocity_j, float(e_rot_j)

        # With probability, perform a purely elastic collision (swap labels).
        # This is momentum- and energy-conserving.
        if self.rng.random() < inelastic_collision_probability:
            return velocity_j.copy(), float(e_rot_j), velocity_i.copy(), float(e_rot_i)

        # Inelastic collision (Borgnakke–Larsen style): redistribute energy between
        # relative translation and rotation, while conserving COM momentum.
        translational_fraction = float(self.rng.random())
        E_rel_post = E_available * translational_fraction
        E_rot_pool_post = E_available - E_rel_post

        # Split rotational pool between particles
        rot_fraction_i = float(self.rng.random())
        new_e_rot_i = E_rot_pool_post * rot_fraction_i
        new_e_rot_j = E_rot_pool_post - new_e_rot_i

        # Sample isotropic relative velocity direction
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

        # Total energy is conserved by construction:
        # E_com + E_rel_post + new_e_rot_i + new_e_rot_j == E_com + E_available.
        return new_velocity_i, float(new_e_rot_i), new_velocity_j, float(new_e_rot_j)
        

        # TODO: add support for different collision models and energy exchange mechanisms such as VHS or VSS
