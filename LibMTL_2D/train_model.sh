#!/bin/bash
python main.py --weighting EW --arch HPS --dataset medmnist-2d --multi_input --save_path log \
    --mode train --download False --epoch 10 --bs 128 --resize True --gpu_id 0