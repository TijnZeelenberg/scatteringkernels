import numpy as np

class borgnakke_larssen_model:

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def postsample(self, velocity_i, velocity_j, m=1.0, T=300.0):
        """
        Perform a collision between two particles using the Borgnakke-Larssen model.
        
        Args:
            velocity_i: Velocity vector of particle i before collision.
            velocity_j: Velocity vector of particle j before collision.
            m: Mass of the particles.
            T: Temperature of the system.
        
        Returns:
            new_velocity_i: Velocity vector of particle i after collision.
            new_velocity_j: Velocity vector of particle j after collision.
        """
        k_B = 1.380649e-23 # Boltzmann constant

        inelastic_collision_probability = 1/245 # From the work of Rabitz and Lam, 1975
        if self.rng.random() < inelastic_collision_probability:
            # Elastic collision: exchange velocities
            return velocity_j, velocity_i
        else:
            # Inelastic collision: randomize velocities based on a distribution
            total_energy = 0.5 * (np.linalg.norm(velocity_i)**2 + np.linalg.norm(velocity_j)**2)
            new_velocity_i = np.random.normal(0, np.sqrt(k_B*T/m), size=velocity_i.shape)
            new_velocity_j = np.random.normal(0, np.sqrt(k_B*T/m), size=velocity_j.shape)

            # Adjust velocities to conserve total energy
            new_total_energy = 0.5 * (np.linalg.norm(new_velocity_i)**2 + np.linalg.norm(new_velocity_j)**2)
            scale_factor = np.sqrt(total_energy / new_total_energy) if new_total_energy > 0 else 1
            new_velocity_i *= scale_factor
            new_velocity_j *= scale_factor

        return new_velocity_i, new_velocity_j
        

        # TODO: add support for different collision models and energy exchange mechanisms such as VHS or VSS