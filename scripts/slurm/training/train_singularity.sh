#!/bin/bash
. ./venv/bin/activate
echo "Running training!"
dir /mnt/

if [ $# -eq 0 ]; then
  # No arguments provided, use default config file
  CONFIG_FILE="configs/network_trainer/train.yml"
else
  # Argument provided, use it as the config file
  CONFIG_FILE="configs/network_trainer/$1"
fi

python main.py --train --config $CONFIG_FILE