
class PlottingConfig:
    def __init__(self):
        self.save_dir = "results"
        self.figsize = (5, 5)

        # label and legend settings
        self.label_fontsize = 12
        self.label_fontweight = 'normal'
        self.legend_fontsize = 10
        self.title_fontsize = 14
        self.title_fontweight = 'bold'

        # scatter plot settings
        self.scatter_point_size = 20
        self.scatter_alpha = 0.7
        self.scatter_cmap = 'viridis'

        # histogram settings
        self.bin_count = 50
        self.hist_density = True
        self.hist_alpha = 0.3