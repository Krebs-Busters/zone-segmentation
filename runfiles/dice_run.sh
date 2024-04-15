#!/bin/bash

for i in 1 2 3 4
do
  echo "dice experiment no: $i "
  PYTHONPATH=. python scripts/train_prostate_X.py \
	  --cost dice \
	  --seed $i \
    --learning-rate 0.00005
done