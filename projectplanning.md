## Timeline

| Phase | Weeks | Focus |
|---|---|---|
| Phase 0 | 1–2 | Literature review |
| Phase 1 | 3–10 | CTC simulations & datasets |
| Phase 2 | 11–20 | ML models & DSMC integration |
| Phase 3 | 21–29 | Validation & thesis writing |


## Roadmap

Phase 0 — Literature Review
- Study rarefied gas dynamics, CTC, and DSMC methods
- Review ML techniques for regression and density estimation

Phase 1 — Data & Simulation
- Generate hydrogen and oxygen collision datasets
- Understand statistical structure of scattering outcomes

Phase 2 — Machine Learning Models
- Train MDN baseline
- Explore alternative architectures
- Integrate ML kernel into DSMC

Phase 3 — Validation & Reporting
- Physical validation (viscosity, bulk viscosity)
- Figures, comparison, thesis writing


## Tasks

### Literature Review
- [x] Review Benjamin Vollebregt's master thesis
- [x] Review Aldo Frezzotti's research
- [x] Review papers on ML for rarefied gas dynamics

### Data & HPC
- [x] Convert Benjamin's CTC code to Python
- [x] Adapt Aldo's code for oxygen collisions
- [x] Generate hydrogen data
- [x] Generate oxygen data
- [ ] Generate large HPC dataset

### Machine Learning
- [x] Implement MDN
- [x] Train MDN baseline
- [x] Implement GMM
- [x] Implement Borgnakke Larsen
- [x] Implement ML scattering kernel in DSMC
- [x] Validate DSMC via viscosity result comparison with SPARTA DSMC.
- [ ] Compare ML scattering kernel with Borgnakke Larsen in DSMC
- [ ] Explore different ML architectures
- [ ] Compare architectures

### Results & Writing
- [ ] Generate figures comparing ML predictions to MD and BL results
- [ ] Write report