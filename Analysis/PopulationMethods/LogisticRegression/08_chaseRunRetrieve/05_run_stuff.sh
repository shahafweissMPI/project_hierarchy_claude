
SCRIPT=./05_find_decoding_neurons_by_exclusion.py
NSTOP=20
MOUSE=afm16924

python -m cuml.accel $SCRIPT mouse=$MOUSE session=240525 n_units_stop=$NSTOP;
sleep 10;

python -m cuml.accel $SCRIPT mouse=$MOUSE session=240527 n_units_stop=$NSTOP;
sleep 10;

python -m cuml.accel $SCRIPT mouse=$MOUSE session=240529 n_units_stop=$NSTOP;
sleep 10;

echo -e "\nAll scripts have finished running!!!\n"
