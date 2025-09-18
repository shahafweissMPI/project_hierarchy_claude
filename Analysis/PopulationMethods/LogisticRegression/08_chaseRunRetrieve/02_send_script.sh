#!/bin/bash

# if you run from a shell, remember to add the lib folder to your PYTHONPATH!!!


export SCRIPT=./02_k_fold_fit_and_save.py

python -m cuml.accel $SCRIPT mouse=afm16924 session=240525;
sleep 10;

python -m cuml.accel $SCRIPT mouse=afm16924 session=240527;
sleep 10;

python -m cuml.accel $SCRIPT mouse=afm16924 session=240529;
sleep 10;

python -m cuml.accel $SCRIPT mouse=afm17365 session=241206;
sleep 10;

python -m cuml.accel $SCRIPT mouse=afm17365 session=241211;
sleep 10;

echo -e "\nAll scripts have finished running!!!\n"
