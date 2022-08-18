#!/bin/bash
python evaluate.py --data_path 'data/test/COL/COL_SMALL/xcsp/*.xml' --model_dir models/COL --num_boost 1 --seed 0 --timeout 1200 --verbose
python evaluate.py --data_path 'data/test/COL/COL_LARGE/xcsp/*.xml' --model_dir models/COL --num_boost 1 --seed 0 --timeout 1200 --verbose
