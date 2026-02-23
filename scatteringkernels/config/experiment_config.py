class ExperimentConfig:
    def __init__(self):

        # Training parameters
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.num_epochs = 100
        self.trainval_split = 0.8
        self.shuffle = True
        self.random_seed = 42

        # Model parameters
        self.num_mixtures = 5
        self.hidden_dim = 128


