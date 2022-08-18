#!/bin/bash
python evaluate.py --data_path 'data/test/RB/*.xml' --model_dir models/RB --num_boost 1 --timeout 1200 --verbose
