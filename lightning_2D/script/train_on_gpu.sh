#!/bin/bash
python main.py --fast_dev_runs 2 --accelerator 'gpu' --devices 1 --max_epoch 10 --batch_size 64 --data_dir /scratch/izar/ishii --tasks all --iter_mode max_size