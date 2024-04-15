#!/bin/bash

for i in 1 2 3 4
do
  echo "sce experiment no: $i "
  PYTHONPATH=. python scripts/train_prostate_X.py \
	  --cost sce \
	  --seed $i
done