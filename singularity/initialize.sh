#!/bin/bash
GPU_ID=""

# Loop through arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu_id) GPU_ID="$2"; shift ;; # Shift twice to get past the value
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift # Shift past the current key (or value, if already shifted above)
done

# Check if GPU_ID is set and is an integer
if [[ -z "$GPU_ID" ]]; then
    echo "Error: --gpu_id argument not provided"
    exit 1
elif ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: --gpu_id must be an integer"
    exit 1
fi

source ./venv/bin/activate
# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPU_ID
