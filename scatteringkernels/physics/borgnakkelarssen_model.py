import numpy as np

class borgnakke_larssen_model:

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def postsample(self, velocity_i, e_rot_i, velocity_j, e_rot_j, m, T=300.0):
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
        inelastic_collision_probability = 1/245 # From the work of Rabitz and Lam, 1975
        if self.rng.random() < inelastic_collision_probability:
            # Elastic collision: exchange velocities and rotational energies
            return velocity_j.copy(), float(e_rot_j), velocity_i.copy(), float(e_rot_i)
        else:
            # Inelastic collision: sample energy split first, then sample directions.
            total_energy = 0.5 * (np.dot(velocity_i, velocity_i) + np.dot(velocity_j, velocity_j)) + e_rot_i + e_rot_j
            if total_energy <= 0.0:
                return np.zeros_like(velocity_i), 0.0, np.zeros_like(velocity_j), 0.0

            # Split total post-collision energy into translational and rotational pools.
            translational_fraction = self.rng.random()
            translational_energy = total_energy * translational_fraction
            rotational_energy = total_energy - translational_energy

            # Split each pool across particles.
            translational_fraction_i = self.rng.random()
            rotational_fraction_i = self.rng.random()
            translational_energy_i = translational_energy * translational_fraction_i
            translational_energy_j = translational_energy - translational_energy_i
            new_e_rot_i = rotational_energy * rotational_fraction_i
            new_e_rot_j = rotational_energy - new_e_rot_i

            # Sample isotropic velocity directions.
            direction_i = self.rng.normal(size=velocity_i.shape)
            direction_j = self.rng.normal(size=velocity_j.shape)

            norm_i = np.sqrt(np.dot(direction_i, direction_i))
            norm_j = np.sqrt(np.dot(direction_j, direction_j))
            if norm_i == 0.0:
                direction_i.fill(0.0)
                direction_i[0] = 1.0
            else:
                direction_i /= norm_i
            if norm_j == 0.0:
                direction_j.fill(0.0)
                direction_j[0] = 1.0
            else:
                direction_j /= norm_j

            speed_i = np.sqrt(2.0 * translational_energy_i / m)
            speed_j = np.sqrt(2.0 * translational_energy_j / m)
            new_velocity_i = direction_i * speed_i
            new_velocity_j = direction_j * speed_j

        return new_velocity_i, new_e_rot_i, new_velocity_j, new_e_rot_j
        

        # TODO: add support for different collision models and energy exchange mechanisms such as VHS or VSS
