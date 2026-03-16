## refactoring

- [x] Refactor 'plot_scattering_comparison()' to take pre-sampled datapoints instead of sampling in the function to reduce the number of sampling calls.
- [x] Create plot functions for scatterplot, histograms that take a config object and data to plot, to further modularize the code and keep styling consistent across different visualizations.
- [x] Write DSMC code
- [ ] Define standard interface for collision models in the DSMC code to allow for easy integration of different collision models (e.g., hard sphere, variable hard sphere, etc.)
- [x] Add support for rotational and vibrational energy modes in the DSMC code, which may require changes to the particle data structure and collision handling logic.
- [x] Implement the 'sample_velocities()' method in the MDN model to generate new velocity vectors based on the predicted energy distribution, which may involve sampling from the predicted mixture of Gaussians and applying the appropriate transformations to obtain velocity vectors.
- [ ] Change DSMC and MDN accuracy from float32 to float64