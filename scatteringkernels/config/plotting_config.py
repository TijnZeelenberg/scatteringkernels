
class PlottingConfig:
    def __init__(self):
        self.save_dir = "plots/"
        self.figsize = (5, 5)

        # label and legend settings
        self.label_fontsize = 12
        self.label_fontweight = 'normal'
        self.legend_fontsize = 10

        # scatter plot settings
        self.scatter_size = 20
        self.scatter_cmap = 'viridis'

        # histogram settings
        self.bin_count = 30
        self.hist_density = True
        self.hist_alpha = 0.7