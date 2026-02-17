## refactoring

- [ ] Refactor 'plot_scattering_comparison()' to take pre-sampled datapoints instead of sampling in the function to reduce the number of sampling calls.
- [ ] Only call gaussian_kde per distribution in visualisations.ipynb and pass the resulting KDE objects to 'plot_scattering_comparison' to avoid redundant KDE computations.
- [ ] Create plot functions for scatterplot, histograms that take an axis object and data to plot, to further modularize the code and keep styling consistent across different visualizations.