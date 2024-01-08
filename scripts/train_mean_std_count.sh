#!/bin/bash

template_file="configs/network_trainer/train_template.yml"
output_file="configs/network_trainer/train_temporary.yml"

combinations=(
    "true true true"
)

# Iterate over the combinations
for combo in "${combinations[@]}"; do
    read use_mean use_std use_count <<< "$combo"
    echo $combo
    # Copy and replace placeholders in the file
    sed -e "s/{use_mean}/$use_mean/g" \
        -e "s/{use_std}/$use_std/g" \
        -e "s/{use_count}/$use_count/g" \
        "$template_file" > "$output_file"

    python main.py --train --config "$output_file"
done