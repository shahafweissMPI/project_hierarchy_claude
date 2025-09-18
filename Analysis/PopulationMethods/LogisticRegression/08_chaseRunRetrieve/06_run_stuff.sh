
SCRIPT=./06_compare_groups.py
MOUSE=afm16924

python -m cuml.accel $SCRIPT mouse=$MOUSE session=240525;
sleep 10;

python -m cuml.accel $SCRIPT mouse=$MOUSE session=240527;
sleep 10;

python -m cuml.accel $SCRIPT mouse=$MOUSE session=240529;
sleep 10;

echo -e "\nAll scripts have finished running!!!\n"
