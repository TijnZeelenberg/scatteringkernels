import matplotlib.pyplot as plt
from config.plotting_config import PlottingConfig
from scipy.stats import gaussian_kde

datasetnames = ["CTC", "MDN", "GMM"]

class Plotter:
    """A class for plotting training histories, scatter plots, and histograms for the scattering kernel project.
    This class uses the settings defined in the PlottingConfig to create consistent and informative visualizations across different datasets and experiments.
    
    Args:
        config (PlottingConfig): An instance of the PlottingConfig class containing plotting settings.
        datasets (dict): A dictionary containing the datasets to be visualized, where keys are dataset names and values 
        are the corresponding data tensors with one column for the x value and one for the y value.
    """
    def __init__(self, config: PlottingConfig):
        self.config = config

    def plot_loss_history(self, train_loss_history, val_loss_history, dataset_name):
        plt.figure(figsize=self.config.figsize)
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch', fontsize=self.config.label_fontsize, fontweight=self.config.label_fontweight)
        plt.ylabel('Loss', fontsize=self.config.label_fontsize, fontweight=self.config.label_fontweight)
        plt.legend(fontsize=self.config.legend_fontsize)
        plt.tight_layout()
        save_path = f"{self.config.save_dir}{dataset_name}_loss_history.png"
        plt.savefig(save_path)
        plt.close()

    def compute_density_per_point(self, x, y):
        data = [x, y]
        kde = gaussian_kde(data)
        return kde(data)
    
    def plot_density_scatter(self, datasets: dict):
        """"Plots a scatter plot of the CTC, MDN and GMM datapoints colored by their density.
        """
        # TODO: use seaborn kdeplot for quicker density estimation and plotting
        figsize_x = self.config.figsize[0] * 3
        figsize_y = self.config.figsize[1] * 2
        fig, ax = plt.subplots(2,3, figsize=(figsize_x, figsize_y))
        for i in range(2):
            x = datasets["inputs"][:,i]
            for j, dataset_name in enumerate(datasetnames):
                y = datasets[dataset_name][:,i]
                density = self.compute_density_per_point(x, y)
                ax[i,j].scatter(x, y, c=density, cmap='viridis', s=self.config.scatter_point_size, alpha=self.config.scatter_alpha)
                ax[i,j].set_xlim(0, 1)
                ax[i,j].set_ylim(0, 1)
                ax[0,j].set_title(f'{dataset_name}', fontsize=self.config.title_fontsize, fontweight=self.config.title_fontweight)
        plt.tight_layout()
        save_path = f"{self.config.save_dir}/plots/density_scatter.png"
        plt.savefig(save_path)
        plt.close()

    def plot_histogram(self, datasets: dict):
        """Plots histograms of the CTC, MDN and GMM datasets for both output variables.""" 
        fig, ax = plt.subplots(1,2, figsize=(self.config.figsize[0]*2, self.config.figsize[1]))
        for i, dataset_name in enumerate(datasetnames):
            ax[0].hist(datasets[dataset_name][:,0], bins=self.config.bin_count, density=self.config.hist_density, alpha=self.config.hist_alpha, label=f'{dataset_name}')
            ax[0].vlines(x=[0.0, 1.0], ymin=0, ymax=ax[0].get_ylim()[1], color='black', linestyle='--', alpha=0.5)
            ax[1].hist(datasets[dataset_name][:,1], bins=self.config.bin_count, density=self.config.hist_density, alpha=self.config.hist_alpha, label=f'{dataset_name}')
            ax[1].vlines(x=[0.0, 1.0], ymin=0, ymax=ax[1].get_ylim()[1], color='black', linestyle='--', alpha=0.5)
            ax[0].set_xlabel('Value', fontsize=self.config.label_fontsize, fontweight=self.config.label_fontweight)
            ax[0].set_ylabel('Density', fontsize=self.config.label_fontsize, fontweight=self.config.label_fontweight)
            ax[1].set_xlabel('Value', fontsize=self.config.label_fontsize, fontweight=self.config.label_fontweight)
            ax[1].set_ylabel('Density', fontsize=self.config.label_fontsize, fontweight=self.config.label_fontweight)
            ax[0].set_title(r'$\epsilon_{tr}$', fontsize=self.config.title_fontsize, fontweight=self.config.title_fontweight)
            ax[1].set_title(r'$\epsilon_{rot,A}$', fontsize=self.config.title_fontsize, fontweight=self.config.title_fontweight)
            ax[0].legend(fontsize=self.config.legend_fontsize)
            ax[1].legend(fontsize=self.config.legend_fontsize)
        plt.tight_layout()
        save_path = f"{self.config.save_dir}/plots/histograms.png"
        plt.savefig(save_path)
        plt.close()
        
    