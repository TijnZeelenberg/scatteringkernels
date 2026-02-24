import matplotlib.pyplot as plt
from config.plotting_config import PlottingConfig
from scipy.stats import gaussian_kde

class Plotter:
    """A class for plotting training histories, scatter plots, and histograms for the scattering kernel project.
    This class uses the settings defined in the PlottingConfig to create consistent and informative visualizations across different datasets and experiments.
    
    Args:
        config (PlottingConfig): An instance of the PlottingConfig class containing plotting settings.
        datasets (dict): A dictionary containing the datasets to be visualized, where keys are dataset names and values 
        are the corresponding data arrays with one column for the x value and one for the y value.
    """
    def __init__(self, config: PlottingConfig, datasets: dict):
        self.config = config
        self.datasets = datasets
        self.datanames = list(datasets.keys())

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
    
    def plot_density_scatter(self):
        """"Plots a scatter plot of the CTC, MDN and GMM datapoints colored by their density.
        Args:
            x (array-like): x-coordinates of the points.
            y (array-like): y-coordinates of the points.
            dataset_name (str): Name of the dataset for saving the plot.
        """
        figsize_x = self.config.figsize[0] * 3
        figsize_y = self.config.figsize[1] * 2
        fig, ax = plt.subplots(2,3, figsize=(figsize_x, figsize_y))
        for i, dataset_name in enumerate(self.datanames):
            x = self.datasets[dataset_name][:,0]
            y = self.datasets[dataset_name][:,1]
            density = self.compute_density_per_point(x, y)
            scatter = ax[i//3, i%3].scatter(x, y, c=density, s=self.config.scatter_size, cmap=self.config.scatter_cmap)
            ax[i//3, i%3].set_xlabel('CTC', fontsize=self.config.label_fontsize, fontweight=self.config.label_fontweight)
            ax[i//3, i%3].set_ylabel('MDN', fontsize=self.config.label_fontsize, fontweight=self.config.label_fontweight)
            ax[i//3, i%3].set_title(dataset_name, fontsize=self.config.label_fontsize+2, fontweight='bold')
            fig.colorbar(scatter, ax=ax[i//3, i%3], label='Density')
        plt.tight_layout()
        save_path = f"{self.config.save_dir}density_scatter.png"
        plt.savefig(save_path)
        plt.close()



    def plot_histogram(self, data):
        for i, dataset_name in enumerate(self.datanames):
            plt.figure(figsize=self.config.figsize)
            plt.hist(data[dataset_name][:,0], bins=self.config.bin_count, density=self.config.hist_density, alpha=self.config.hist_alpha, label='CTC')
            plt.hist(data[dataset_name][:,1], bins=self.config.bin_count, density=self.config.hist_density, alpha=self.config.hist_alpha, label='MDN')
            plt.xlabel('Value', fontsize=self.config.label_fontsize, fontweight=self.config.label_fontweight)
            plt.ylabel('Density', fontsize=self.config.label_fontsize, fontweight=self.config.label_fontweight)
            plt.title(f'Histogram of {dataset_name}', fontsize=self.config.label_fontsize+2, fontweight='bold')
            plt.legend(fontsize=self.config.legend_fontsize)
            plt.tight_layout()
            save_path = f"{self.config.save_dir}{dataset_name}_histogram.png"
            plt.savefig(save_path)
            plt.close()