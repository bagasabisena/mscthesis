#!/usr/bin/env bash

time python synthetic.py synthetic_2_0_1 dummy -n dummy -p run_greedy -o 5 10 20 -m 20 -w 1 -e '{"max_kernel": 2, "max_mult": 2}'
time python synthetic.py synthetic_2_0_1 dummy -n dummy -p run_exhaustive -o 5 10 20 -m 20 -w 1 -e '{"max_kernel": 2, "max_mult": 2}'
time python synthetic.py synthetic_0_1 dummy -n dummy -p run_greedy -o 5 10 20 -m 20 -w 1 -e '{"max_kernel": 4, "max_mult": 2}'
time python synthetic.py synthetic_0_1 dummy -n dummy -p run_exhaustive -o 5 10 20 -m 20 -w 1 -e '{"max_kernel": 4, "max_mult": 2}'
