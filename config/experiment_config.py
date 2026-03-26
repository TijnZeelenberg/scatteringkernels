from typing import Literal

CovarianceType = Literal['full', 'tied', 'diag', 'spherical']

class ExperimentConfig:
    def __init__(self):
        
        # MDN settings
        # Training parameters
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.num_epochs = 100
        self.trainval_split = 0.8
        self.shuffle = True
        self.random_seed = 42

        # Model parameters
        self.input_dim = 3
        self.output_dim = 2
        self.num_mixtures = 5
        self.hidden_dim = 128
        
        # Dataset parameters
        self.num_samples = 10000

        # GMM settings
        self.gmm_n_components = 5
        self.gmm_covariance_type:CovarianceType = 'full'


