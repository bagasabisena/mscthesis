#!/usr/bin/env bash

python synthetic.py osynthetic_2_0_1 blr20_osynthetic_2_0_1 -n blr -p run_bayesian_lr -o 5 10 20  -w 20 -e '{"lag": 5}'
python synthetic.py osynthetic_0_1 blr20_osynthetic_0_1 -n blr -p run_bayesian_lr -o 5 10 20 -w 20 -e '{"lag": 12}'