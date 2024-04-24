#!/bin/bash
python main.py --precision '32' --accelerator 'gpu' --devices 1 --max_epochs 40 --batch_size 96 --data_dir /scratch/izar/ishii --tasks pathmnist --iter_mode max_size