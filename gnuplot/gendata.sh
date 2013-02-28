#!/bin/bash
rm plot.dat
for i in {1..30}; do
    echo -e "$i\t$((i*i))" >> plot.dat
    sleep 1
    echo "done"
done