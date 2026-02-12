### DSMC Simulation steps
    1. Initialization: The simulation domain is defined, and the initial conditions for the gas particles are set. This includes specifying the number of particles, their positions, velocities, the box size, cell size, and the time step for the simulation.
    2. Collision Detection: The code identifies which particles are in the same cell and therefore have the potential to collide. This is typically done by dividing the simulation domain into a grid of cells and checking which particles are located in each cell.
    3. Collision Handling: For each cell, the code calculates the probability of collisions occurring between particles based on their relative velocities and the collision cross-section. If a collision is determined to occur, the code uses a collision model (such as Borgnakke-Larsen or an ML model) to update the velocities of the colliding particles according to the conservation of momentum and energy.
    4. Particle Movement: After handling collisions, the code updates the positions of all particles based on their velocities and the time step. This involves moving each particle according to its velocity and checking for any boundary conditions (e.g., reflecting or periodic boundaries).
    5. Data Collection: The code collects data on various properties of the gas, such as temperature, pressure, and velocity distributions. This data can be used for analysis and visualization of the simulation results.
    6. Iteration: The code repeats steps 2-5 for a specified number of time steps or until a certain condition is met (e.g., reaching a steady state). This iterative process allows the simulation to evolve over time and capture the dynamics of the gas particles under the specified conditions.
    7. Finalization: Once the simulation is complete, the code may perform any necessary cleanup tasks, such as saving the final state of the particles, generating output files for analysis, and releasing any allocated resources.

## Code structure
The DSMC simulation code will be organized under a single class called `DSMC_Simulation`, which will encapsulate all the necessary functions and data structures for running the simulation. This class will include methods for each of the steps outlined above, as well as any additional helper functions needed for specific tasks (e.g., calculating collision probabilities, handling boundary conditions, etc.). The class will also include attributes to store the properties of the particles and the simulation domain.

### Data structures
- 'positions' : np.array of shape (N, 3) to store the x, y, z coordinates of each particle.
- 'velocities' : np.array of shape (N, 3) to store the velocity components (vx, vy, vz) of each particle.
- 'cell_indices' : np.array of shape (N,) to store the index of the cell each particle belongs to.

### Required methods
    - 'initialize_domain()': Sets up the simulation domain, including defining the box size, cell size, and time step.
    - `initialize_particles()`: Initializes the positions and velocities of the particles based on the specified initial conditions.
    - 'bin_particles()': Organizes particles into cells based on their positions to facilitate collision detection.
    - `select_collisionpairs()`: Identifies which particles are in the same cell and calculates the probability of collisions occurring.
    - `perform_collisions()`: Updates the velocities of colliding particles according to the chosen collision model.
    - `move_particles()`: Updates the positions of all particles based on their velocities and the time step, while checking for boundary conditions.
    - `apply_boundaries()`: Checks for any boundary conditions and updates particle positions accordingly (e.g., reflecting or periodic boundaries).
    - `sample_macroscopic_properties()`: Collects and stores data on various properties of the gas for analysis and visualization.
    - `simulate()`: Orchestrates the overall simulation process by calling the necessary functions in the correct sequence for a specified number of time steps.
    
    ### Optional methods
    - `save_state()`: Saves the current state of the particles to a file for later analysis or visualization.
    - `load_state()`: Loads a previously saved state of the particles to continue a simulation or analyze past results.
