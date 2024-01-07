#!/bin/bash

for N in 1
do
    # Copy the file, replacing {N} with the value of N
    sed "s/{N}/$N/g" configs/data_generator/coco.yml > configs/data_generator/coco_temp.yml

    for i in {1..5}
    do
        python main.py --generate_data --config configs/data_generator/coco_temp.yml
    done
done
