#!/usr/bin/env bash

python synthetic.py synthetic_0_1 result_synth_01_ex -n exhaustive -p run_exhaustive -o 5 10 20  > result_synth_01_ex.out 2> result_synth_01_ex.err
python synthetic.py synthetic_1 result_synth_1_ex -n exhaustive -p run_exhaustive -o 5 10 20 > result_synth_1_ex.txt 2> result_synth_1_ex.err
python synthetic.py synthetic_10 result_synth_10_ex -n exhaustive -p run_exhaustive -o 5 10 20 > result_synth_10_ex.txt 2> result_synth_10_ex.err
