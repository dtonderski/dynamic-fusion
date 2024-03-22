#!/bin/bash

# Check for the --test flag
test_mode=false
while [ "$#" -gt 0 ]; do
    case $1 in
        --test) test_mode=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Activate virtual environment
. ./venv/bin/activate
echo "Running data generation!"

# Determine the config file based on test_mode
if $test_mode; then
    config_file="configs/data_generator/coco_test.yml"
else
    config_file="configs/data_generator/coco.yml"
fi

# Execute the Python command with the chosen config
python main.py --generate_data --config $config_file
