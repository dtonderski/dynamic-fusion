#!/bin/bash

for N in 1
do
    # Copy the file, replacing {N} with the value of N
    sed "s/{N}/$N/g" configs/network_trainer/train.yml > configs/network_trainer/train_temp.yml

    python main.py --train --config configs/network_trainer/train_temp.yml
done
