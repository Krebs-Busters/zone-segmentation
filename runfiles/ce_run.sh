#!/bin/bash

for i in 1 2 3 4
do
  echo "ce experiment no: $i "
  PYTHONPATH=. python scripts/train_prostate_X.py \
	  --cost ce \
	  --seed $i
done