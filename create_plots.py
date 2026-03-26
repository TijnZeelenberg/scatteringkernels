from visualization.plot import Plotter
from utils.helpers import load_dataset
from machinelearning.mdn_model import MixtureDensityNetwork
from machinelearning.gmm_model import GaussianMixtureModel
from config.experiment_config import ExperimentConfig 
from config.plotting_config import PlottingConfig
from analysis.kl_divergence import kl_divergence

config = ExperimentConfig()

# Load CTC dataset
ctc_data = load_dataset("data/O2O2_collisions.csv")

# Sample MDN
mdn = MixtureDensityNetwork(input_dim=config.input_dim, output_dim=config.output_dim, num_mixtures=config.num_mixtures, hidden_dim=config.hidden_dim) 
mdn.load_model("results/models/mdn_O2O2.pth")
mdn_samples = mdn.sample(x=ctc_data[0])

# Sample GMM
gmm = GaussianMixtureModel(n_components=config.gmm_n_components, covariance_type=config.gmm_covariance_type)
gmm.fit(ctc_data[1])
gmm_samples = gmm.sample(num_samples=config.num_samples)

datasets = {
    "inputs": ctc_data[0][:,1:], # Use only the energy fractions for plotting not the total energy
    "CTC": ctc_data[1],
    "MDN": mdn_samples,
    "GMM": gmm_samples
}

# Plotting
plotting_config = PlottingConfig()
plotter = Plotter(config=plotting_config)
plotter.plot_density_scatter(datasets=datasets)
plotter.plot_histogram(datasets=datasets)

# Compute KL divergence 
kl_ctc_mdn = kl_divergence(datasets["CTC"][0], datasets["MDN"][0])
kl_ctc_gmm = kl_divergence(datasets["CTC"][0], datasets["GMM"][0])
print(f"KL Divergence between CTC and MDN: {kl_ctc_mdn:.4f}")
print(f"KL Divergence between CTC and GMM: {kl_ctc_gmm:.4f}")