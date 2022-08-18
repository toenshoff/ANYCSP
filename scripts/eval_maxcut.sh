#!/bin/bash
python evaluate_maxcut.py --data_path 'data/test/GSET/G43.mtx' --model_dir models/MAXCUT --num_boost 20 --seed 0 --timeout 180 --verbose
