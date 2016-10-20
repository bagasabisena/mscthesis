#!/usr/bin/env bash

python synthetic.py synthetic_2_0_1 result_synth_2_01_auto -n auto2 -p run_auto_arima -o 5 10 20  > result_synth_2_01_auto.txt 2> result_synth_2_01_auto.err
python synthetic.py synthetic_2_1 result_synth_2_1_auto -n auto2 -p run_auto_arima -o 5 10 20 > result_synth_2_1_auto.txt 2> result_synth_2_1_auto.err
python synthetic.py synthetic_2_10 result_synth_2_10_auto -n auto2 -p run_auto_arima -o 5 10 20 > result_synth_2_10_auto.txt 2> result_synth_2_10_auto.err

python synthetic.py synthetic_2_0_1 result_synth_2_01_nar -n nar2 -p run_narx -o 5 10 20  > result_synth_2_01_nar.txt 2> result_synth_2_01_nar.err
python synthetic.py synthetic_2_1 result_synth_2_1_nar -n nar2 -p run_narx -o 5 10 20 > result_synth_2_1_nar.txt 2> result_synth_2_1_nar.err
python synthetic.py synthetic_2_10 result_synth_2_10_nar -n nar2 -p run_narx -o 5 10 20 > result_synth_2_10_nar.txt 2> result_synth_2_10_nar.err

python synthetic.py synthetic_2_0_1 result_synth_2_01_lr -n lr2 -p run_linear_regression -o 5 10 20  > result_synth_2_01_lr.txt 2> result_synth_2_01_lr.err
python synthetic.py synthetic_2_1 result_synth_2_1_lr -n lr2 -p run_linear_regression -o 5 10 20 > result_synth_2_1_lr.txt 2> result_synth_2_1_lr.err
python synthetic.py synthetic_2_10 result_synth_2_10_lr -n lr2 -p run_linear_regression -o 5 10 20 > result_synth_2_10_lr.txt 2> result_synth_2_10_lr.err