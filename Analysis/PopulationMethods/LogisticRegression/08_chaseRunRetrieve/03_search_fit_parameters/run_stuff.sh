#!/bin/bash
# if you run from a shell, remember to add the lib folder to your PYTHONPATH!!!

SCRIPT=fit_score_only.py
dt_all=(10E-3 20E-3)
penalty_all=('l1' 'l2')
C_all=(0.01 0.1 1.0 10.0)

MOUSE=afm16924
SESSION="240525"

echo -e "Fitting mouse $MOUSE session $SESSION\n"

for dt in "${dt_all[@]}"; do
  for penalty in "${penalty_all[@]}"; do
    for C in "${C_all[@]}"; do
      python -m cuml.accel $SCRIPT with\
         the_mouse=$MOUSE the_session=$SESSION\
         dt=$dt penalty=$penalty C_regression=$C;
      sleep 5;
      #echo -e "Test completed!!!"
      #exit 0 # just for testing
    done
  done
done

echo -e "Fitting completed for mouse $MOUSE session $SESSION\n"


SESSION="240527"

echo -e "Fitting mouse $MOUSE session $SESSION\n"

for dt in "${dt_all[@]}"; do
  for penalty in "${penalty_all[@]}"; do
    for C in "${C_all[@]}"; do
      python -m cuml.accel $SCRIPT with\
         the_mouse=$MOUSE the_session=$SESSION dt=$dt penalty=$penalty C_regression=$C;
      sleep 5;
    done
  done
done

echo -e "Fitting completed for mouse $MOUSE session $SESSION\n"

SESSION="240529"

echo -e "Fitting mouse $MOUSE session $SESSION\n"

for dt in "${dt_all[@]}"; do
  for penalty in "${penalty_all[@]}"; do
    for C in "${C_all[@]}"; do
      python -m cuml.accel $SCRIPT with\
         the_mouse=$MOUSE the_session=$SESSION dt=$dt penalty=$penalty C_regression=$C;
      sleep 5;
    done
  done
done

echo -e "Fitting completed for mouse $MOUSE session $SESSION\n"

MOUSE=afm17365
SESSION="241206"

echo -e "Fitting mouse $MOUSE session $SESSION\n"

for dt in "${dt_all[@]}"; do
  for penalty in "${penalty_all[@]}"; do
    for C in "${C_all[@]}"; do
      python -m cuml.accel $SCRIPT with\
         the_mouse=$MOUSE the_session=$SESSION dt=$dt penalty=$penalty C_regression=$C;
      sleep 5;
    done
  done
done

echo -e "Fitting completed for mouse $MOUSE session $SESSION\n"

SESSION="241211"

echo -e "Fitting mouse $MOUSE session $SESSION\n"

for dt in "${dt_all[@]}"; do
  for penalty in "${penalty_all[@]}"; do
    for C in "${C_all[@]}"; do
      python -m cuml.accel $SCRIPT with\
         the_mouse=$MOUSE the_session=$SESSION dt=$dt penalty=$penalty C_regression=$C;
      sleep 5;
    done
  done
done

echo -e "Fitting completed for mouse $MOUSE session $SESSION\n"
echo -e "All sessions completed !!! "