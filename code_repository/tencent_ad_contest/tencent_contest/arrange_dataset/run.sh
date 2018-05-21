#! /bin/bash

echo 'convert: userFeature -> userFeature.csv ...'
# python3 vw2csv.py

echo 'assemble: train.csv, adFeature.csv, userFeature -> train_assemble.csv ...'
echo 'assemble: test1.csv, adFeature.csv, userFeature -> test_assemble.csv ...'
python3 make_one.py

echo 'convert: train_assemble.csv -> train_full.csv ...'
python3 make_train.py

echo 'convert: test_assemble.csv -> pred_full.csv ...'
python3 make_pred.py

echo 'arrange dataset finished! good luck ^.^'