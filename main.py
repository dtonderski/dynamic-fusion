import argparse
import logging
from pathlib import Path
from typing import Any, List, Tuple

from pydantic import BaseModel
from ruamel.yaml import YAML

from dynamic_fusion.data_generator import DataGenerator, DataGeneratorConfiguration
from dynamic_fusion.data_generator.configuration import EventDiscretizerConfiguration
from dynamic_fusion.data_generator.standalone_discretizer import StandaloneDiscretizer
from dynamic_fusion.data_generator.threshold_adder import ThresholdAdder
from dynamic_fusion.interactive_visualizer import Visualizer, VisualizerConfiguration
from dynamic_fusion.network_trainer import Trainer, TrainerConfiguration


def generate_data_reconstruction(arguments: argparse.Namespace, unknown_args: List[str]) -> None:
    with open(arguments.config, encoding="utf8") as infile:
        yaml = YAML().load(infile)
        config = DataGeneratorConfiguration.parse_obj(yaml)
        update_config(config, unknown_args)
    data_generator = DataGenerator(config)
    data_generator.run()


def train_reconstruction(arguments: argparse.Namespace, unknown_args: List[str]) -> None:
    with open(arguments.config, encoding="utf8") as infile:
        yaml = YAML().load(infile)
        config = TrainerConfiguration.parse_obj(yaml)
        update_config(config, unknown_args)
    network_trainer = Trainer(config)
    network_trainer.run()


def visualize_interactive(arguments: argparse.Namespace, unknown_args: List[str]) -> None:
    with open(arguments.config, encoding="utf8") as infile:
        yaml = YAML().load(infile)
        config = VisualizerConfiguration.parse_obj(yaml)
        update_config(config, unknown_args)
    network_trainer = Visualizer(config)
    network_trainer.run()


def add_thresholds(arguments: argparse.Namespace, unknown_args: List[str]) -> None:
    with open(arguments.config, encoding="utf8") as infile:
        yaml = YAML().load(infile)
        config = DataGeneratorConfiguration.parse_obj(yaml)
        update_config(config, unknown_args)
    threshold_adder = ThresholdAdder(arguments.thresholds, config)
    threshold_adder.run()


def discretize_thresholds(arguments: argparse.Namespace, unknown_args: List[str]) -> None:
    with open(arguments.config, encoding="utf8") as infile:
        yaml = YAML().load(infile)
        config = DataGeneratorConfiguration.parse_obj(yaml)
        update_config(config, unknown_args)

    if arguments.discretizer_config == "original":
        discretizer_config = None
    else:
        with open(arguments.config, encoding="utf8") as infile:
            yaml = YAML().load(infile)
            discretizer_config = EventDiscretizerConfiguration.parse_obj(yaml)

    standalone_discretizer = StandaloneDiscretizer(arguments.thresholds, config, arguments.allow_overwrite, discretizer_config)
    standalone_discretizer.run()


def update_config(config: BaseModel, unknown_args: List[str]) -> None:
    for arg in unknown_args:
        if not arg.startswith("--"):
            continue
        if not "=" in arg:
            continue
        key, val = arg[2:].split("=")
        keys = key.split(".")
        update_nested_config(config, keys, val)


def update_nested_config(config: BaseModel, key_list: List[str], value: Any) -> None:
    """Recursively update nested dictionary."""
    key = key_list[0]
    if len(key_list) == 1:
        if isinstance(getattr(config, key), Path):
            value = Path(value)
            print(value)
        setattr(config, key, value)
    else:
        if key not in config.__fields__.keys() or not isinstance(getattr(config, key), BaseModel):
            raise ValueError(f"Error setting {key_list} to {value}")
        update_nested_config(getattr(config, key), key_list[1:], value)


def main(arguments: argparse.Namespace, unknown_args: List[str]) -> None:
    logging.basicConfig(level=logging.INFO)
    if arguments.generate_data:
        return generate_data_reconstruction(arguments, unknown_args)
    if arguments.train:
        return train_reconstruction(arguments, unknown_args)
    if arguments.visualize:
        return visualize_interactive(arguments, unknown_args)
    if arguments.add_thresholds:
        return add_thresholds(arguments, unknown_args)
    if arguments.discretize_thresholds:
        return discretize_thresholds(arguments, unknown_args)
    raise NotImplementedError()


def parse_arguments() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--generate_data", action="store_true", help="generate data")
    mode_group.add_argument("--train", action="store_true", help="train the model")
    mode_group.add_argument("--visualize", action="store_true", help="interactively visualize model")
    mode_group.add_argument("--add_thresholds", action="store_true", help="regenerate events with given thresholds using data existing transforms")
    mode_group.add_argument("--discretize_thresholds", action="store_true", help="(re-)discretize events for given thresholds")

    parser.add_argument("--thresholds", type=float, nargs="+", help="List of thresholds", default=[])
    parser.add_argument(
        "--discretizer_config", required=False, type=str, help="path to new discretizer configuration, set to 'original' to use original config"
    )
    parser.add_argument(
        "--allow_overwrite", required=False, type=bool, default=False, help="used with --discretize_thresholds to set OK to overwrite discretized events"
    )

    parsed_args, unknown_args = parser.parse_known_args()

    if (parsed_args.add_thresholds or parsed_args.discretize_thresholds) and not parsed_args.thresholds:
        parser.error("--add_thresholds and --discretize_thresholds require --threshold to be set with a list of floats.")

    if parsed_args.discretize_thresholds and not parsed_args.discretizer_config:
        parser.error("--discretize_thresholds requires --discretizer_config to be set. Set to original if you want to use 'original' config.")

    return parsed_args, unknown_args


if __name__ == "__main__":
    args, unknown = parse_arguments()
    main(args, unknown)
