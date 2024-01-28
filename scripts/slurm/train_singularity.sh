#!/bin/bash
. ./venv/bin/activate
echo "Running training!"
python main.py --train --config configs/network_trainer/train.yml --data_handler.dataset.dataset_directory=/mnt