#!/usr/bin/env bash

python synthetic.py synthetic_2_0_1 k2_10 -n true -p run_true_kernel -o 5 10 20 -e '{"model": 2}'
python synthetic.py synthetic_2_1 k2_1 -n true -p run_true_kernel -o 5 10 20 -e '{"model": 2}'
python synthetic.py synthetic_1 k4_10 -n true -p run_true_kernel -o 5 10 20 -e '{"model": 4}'
python synthetic.py synthetic_10 k4_1 -n true -p run_true_kernel -o 5 10 20 -e '{"model": 4}'