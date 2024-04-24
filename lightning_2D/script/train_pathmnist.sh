#!/bin/bash
python main.py --log_every_n_steps 10 --precision '32' --accelerator 'gpu' --devices 1 --max_epochs 50 --batch_size 96 --data_dir /scratch/izar/ishii --tasks pathmnist --iter_mode max_size