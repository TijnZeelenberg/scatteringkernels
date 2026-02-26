## refactoring

- [x] Refactor 'plot_scattering_comparison()' to take pre-sampled datapoints instead of sampling in the function to reduce the number of sampling calls.
- [x] Create plot functions for scatterplot, histograms that take a config object and data to plot, to further modularize the code and keep styling consistent across different visualizations.
- [ ] Write DSMC code