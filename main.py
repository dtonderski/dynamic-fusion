import argparse
import logging

from ruamel.yaml import YAML


def generate_data_reconstruction(arguments: argparse.Namespace) -> None:
    # pylint: disable-next=import-outside-toplevel
    from dynamic_fusion.data_generator import (
        DataGeneratorConfiguration,
        DataGenerator,
    )

    logging.basicConfig(level=logging.INFO)
    with open(arguments.config, encoding="utf8") as infile:
        yaml = YAML().load(infile)
        config = DataGeneratorConfiguration.parse_obj(yaml)
    data_generator = DataGenerator(config)
    data_generator.run()


def main(arguments: argparse.Namespace) -> None:
    generate_data_reconstruction(arguments)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
