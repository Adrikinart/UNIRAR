#!/bin/bash

model_list=("Unisal" "TranSalNetRes" "TranSalNetDense")

for model_name in "${model_list[@]}"
do
    threshold_list=(0.1 0.5 0.9)

    for threshold in "${threshold_list[@]}"
    do
        python run_eval_rarity_network.py --model "$model_name" --threshold "$threshold"
    done
done
