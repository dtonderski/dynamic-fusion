import argparse
import logging

from ruamel.yaml import YAML

from dynamic_fusion.data_generator import DataGenerator, DataGeneratorConfiguration
from dynamic_fusion.network_trainer import Trainer, TrainerConfiguration
from dynamic_fusion.utils.seeds import set_seeds

def generate_data_reconstruction(arguments: argparse.Namespace) -> None:
    # pylint: disable-next=import-outside-toplevel
    with open(arguments.config, encoding="utf8") as infile:
        yaml = YAML().load(infile)
        config = DataGeneratorConfiguration.parse_obj(yaml)
    data_generator = DataGenerator(config)
    data_generator.run()


def train_reconstruction(arguments: argparse.Namespace) -> None:
    with open(arguments.config, encoding="utf8") as infile:
        yaml = YAML().load(infile)
        config = TrainerConfiguration.parse_obj(yaml)
    network_trainer = Trainer(config)
    network_trainer.run()


def main(arguments: argparse.Namespace) -> None:
    if arguments.seed is not None:
        set_seeds(arguments.seed)
    logging.basicConfig(level=logging.INFO)

    if arguments.generate_data:
        return generate_data_reconstruction(arguments)
    if arguments.train:
        return train_reconstruction(arguments)
    raise NotImplementedError()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", required=False, default=0, type=int)
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--generate_data", action="store_true", help="generate data"
    )
    mode_group.add_argument("--train", action="store_true", help="train the model")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
