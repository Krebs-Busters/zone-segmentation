#!/bin/bash

for i in 1 2 3 4
do
  echo "sigfocal experiment no: $i "
  PYTHONPATH=. python scripts/train_prostate_X.py \
	  --cost sigfocal \
	  --seed $i \
    --gamma 1.5
done