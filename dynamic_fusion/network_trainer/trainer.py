import torch

from dynamic_fusion.utils.seeds import set_seeds
from .configuration import TrainerConfiguration
from .data_handler import DataHandler
from .network_fitter import NetworkFitter
from .network_loader import NetworkLoader
from .training_monitor import TrainingMonitor


class Trainer:
    config: TrainerConfiguration
    data_handler: DataHandler
    network_loader: NetworkLoader
    network_fitter: NetworkFitter

    def __init__(self, config: TrainerConfiguration) -> None:
        torch.multiprocessing.set_start_method("spawn")
        self.config = config
        if self.config.seed is not None:
            set_seeds(self.config.seed)
        self.data_handler = DataHandler(config.data_handler, config.shared)
        self.network_loader = NetworkLoader(config.network_loader, config.shared)
        self.training_monitor = TrainingMonitor(config)
        self.network_fitter = NetworkFitter(self.training_monitor, config.network_fitter, config.shared)

    def run(self) -> None:
        data_loader = self.data_handler.run()
        reconstruction_network, decoding_network = self.network_loader.run()
        self.network_fitter.run(data_loader, reconstruction_network, decoding_network)
