#!/bin/bash
python evaluate.py --data_path 'data/test/MAXSAT/3CNF/*.cnf' --model_dir models/MAXSAT --num_boost 1 --timeout 1200 --verbose
python evaluate.py --data_path 'data/test/MAXSAT/4CNF/*.cnf' --model_dir models/MAXSAT --num_boost 1 --timeout 1200 --verbose
python evaluate.py --data_path 'data/test/MAXSAT/5CNF/*.cnf' --model_dir models/MAXSAT --num_boost 1 --timeout 1200 --verbose
