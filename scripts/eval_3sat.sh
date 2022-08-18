#!/bin/bash
python evaluate.py --data_path 'data/test/SATLIB/UF50*/*.cnf' --model_dir models/3SAT --num_boost 10 --network_steps 10000
python evaluate.py --data_path 'data/test/SATLIB/UF100*/*.cnf' --model_dir models/3SAT --num_boost 10 --network_steps 10000
python evaluate.py --data_path 'data/test/SATLIB/UF150*/*.cnf' --model_dir models/3SAT --num_boost 10 --network_steps 10000
python evaluate.py --data_path 'data/test/SATLIB/UF200*/*.cnf' --model_dir models/3SAT --num_boost 10 --network_steps 10000
python evaluate.py --data_path 'data/test/SATLIB/UF250*/*.cnf' --model_dir models/3SAT --num_boost 10 --network_steps 10000
