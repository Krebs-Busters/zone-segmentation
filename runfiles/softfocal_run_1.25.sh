#!/bin/bash

for i in 1 2 3 4
do
  echo "softfocal experiment no: $i "
  PYTHONPATH=. python scripts/train_prostate_X.py \
	  --cost softfocal \
	  --seed $i \
    --gamma 1.25
done