#! /bin/bash

for i in $(seq 20); do
    val=`expr $i - 1`
    nohup ./csv2libffm.py --index $val >"$val.log" 2>&1 & 
done