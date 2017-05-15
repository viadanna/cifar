#!/bin/bash

##
# Run all models with three different training dataset sizes.
##

for A in 40000 120000 400000; do
    for M in simple simple_reg lenet lenet_reg deep deeper mininception; do
        python2 experiment.py $M -a $A
    done
done
