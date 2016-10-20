#!/usr/bin/env bash

python synthetic.py osynthetic_2_0_1 greedy40_osynthetic_2_0_1 -n greedy -p run_greedy -o 5 10 20 40 -m 40 -w 20 -e '{"max_kernel": 2}' > result_synth_01_greedy.out 2> result_synth_01_greedy.err
python synthetic.py osynthetic_0_1 greedy40_osynthetic_0_1 -n greedy -p run_greedy -o 5 10 20 40 -m 40 -w 20 -e '{"max_kernel": 4}' > result_synth_1_greedy.txt 2> result_synth_1_greedy.err
# python synth_run.py