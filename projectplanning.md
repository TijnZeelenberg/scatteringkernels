## Milestones
- [x] Literature Review on Rarefied gas dynamics
- [x] Convert Benjamin Vollebregt's CTC code to Python and generate hydrogen collision data
- [x] Adapt Aldo Frezzotti's code to generate oxygen collision data
- [x] Explore and visualize collision data to understand underlying patterns
- [x] Create MDN model for energy fraction prediction
- [x] Train MDN model on hydrogen and oxygen collision data
- [ ] Generate large training dataset for hydrogen and oxygen collisions on HPC cluster
- [ ] Implement ML scattering kernel in Sven Bendermacher's DSMC code
- [ ] Validate ML scattering kernel by predicting viscosity and bulk viscosity.
- [ ] Come up with different ML architectures for scattering kernel prediction (e.g. normalizing flows, GANs, etc.)
- [ ] Compare different ML architectures for scattering kernel prediction 
- [ ] Generate figures comparing ML scattering kernel predictions to CTC results for in report
- [ ] Write report on results and future work

## Project Planning 
| Task | Start date | End date | Status |
|-----|------|--------|-------|
| Phase 0 | Week 1 | Week 2 | Finished |
| Literature review | | | ✅ |
| Phase 1: CTC Simulation | Week 3 | Week 10 | Finished |
| Research  CTC & DSMC | | | ✅|
| Set up CTC simulations  | | | ✅|
| Generate hydrogen collision data | | | ✅|
| Generate oxygen collision data | | | ✅|
| Explore data | | | ✅|
| Phase 2: Machine Learning | Week 11 | Week 20 | |
| Set up MDN model | | | ✅|
| Train MDN model | | | ✅| 
| Generate training data on HPC cluster | | | |
| Implement ML scattering kernel in DSMC code | | | |
| Validate ML scattering kernel in DSMC | | | |
| Create different ML architectures | | | |
| Compare ML algorithms | | | |
| Phase 3: Results | Week 21 | Week 29 | |
| Dive back into the literature | | | |
| Generate useful figures | | | |
| Write report | | | |
