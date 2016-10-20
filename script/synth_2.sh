#!/usr/bin/env bash

python synthetic.py synthetic_2_0_1 result_synth_2_01_greedy -n greedy2 -p run_greedy -o 5 10 20  > result_synth_2_01_greedy.out 2> result_synth_2_01_greedy.err
python synthetic.py synthetic_2_1 result_synth_2_1_greedy -n greedy2 -p run_greedy -o 5 10 20 > result_synth_2_1_greedy.txt 2> result_synth_2_1_greedy.err
python synthetic.py synthetic_2_10 result_synth_2_10_greedy -n greedy2 -p run_greedy -o 5 10 20 > result_synth_2_10_greedy.txt 2> result_synth_2_10_greedy.err

python synthetic.py synthetic_2_0_1 result_synth_2_01_ex -n exhaustive2 -p run_exhaustive -o 5 10 20  > result_synth_2_01_ex.out 2> result_synth_2_01_ex.err
python synthetic.py synthetic_2_1 result_synth_2_1_ex -n exhaustive2 -p run_exhaustive -o 5 10 20 > result_synth_2_1_ex.txt 2> result_synth_2_1_ex.err
python synthetic.py synthetic_2_10 result_synth_2_10_ex -n exhaustive2 -p run_exhaustive -o 5 10 20 > result_synth_2_10_ex.txt 2> result_synth_2_10_ex.err
