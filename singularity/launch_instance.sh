#!/bin/bash
# Check if the second argument is provided for the instance name
if [ $# -gt 0 ]; then
    INSTANCE_NAME=$2
else
    # Define a default instance name or generate it dynamically
    INSTANCE_NAME="default_instance_name"
fi
echo Starting instance with name $INSTANCE_NAME
singularity instance start --bind .:/mnt --nv singularity/python39.sif $INSTANCE_NAME
