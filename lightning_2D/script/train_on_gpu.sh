#!/bin/bash
python main.py --accelerator 'gpu' --devices 1 --max_epochs 20 --batch_size 96 --data_dir /scratch/izar/ishii --tasks all --iter_mode max_size