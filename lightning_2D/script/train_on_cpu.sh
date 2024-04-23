#!/bin/bash
python main.py --fast_dev_runs 10 --accelerator 'cpu' --max_epoch 10 --batch_size 32 --data_dir /scratch/izar/ishii --tasks all --iter_mode max_size