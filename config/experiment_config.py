from typing import Literal

CovarianceType = Literal["full", "tied", "diag", "spherical"]


class ExperimentConfig:
    def __init__(self):

        # MDN settings
        # Training parameters
        self.learning_rate = 2.71e-04
        self.batch_size = 256
        self.num_epochs = 200
        self.trainval_split = 0.7
        self.shuffle = True
        self.random_seed = 41

        # Model parameters
        self.input_dim = 3
        self.output_dim = 2
        self.num_mixtures = 24
        self.hidden_dim = 256
        self.dropout = 0.132

        # Dataset parameters
        self.num_samples = 10000

        # GMM settings
        self.gmm_n_components = 5
        self.gmm_covariance_type: CovarianceType = "full"
