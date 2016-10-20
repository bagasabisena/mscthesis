#!/usr/bin/env bash

python synthetic.py osynthetic_2_0_1 persistence40_osynthetic_2_0_1 -n persistence -p run_persistence -o 5 10 20 40 -w 20 -m 40
python synthetic.py osynthetic_0_1 persistence40_osynthetic_0_1 -n persistence -p run_persistence -o 5 10 20 40 -w 20 -m 40