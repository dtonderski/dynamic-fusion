#!/bin/bash


# Activate virtual environment
. ./venv/bin/activate

echo "Running E2VID test"

# Execute the Python command with the chosen config
python dynamic_fusion/scripts/test_e2vid_data.py