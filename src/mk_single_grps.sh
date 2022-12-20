#!/bin/bash

if [ ! -d "./single-split-train-data" ];
    then
    echo 'folder single-split-train-data does not exist!'
    mkdir 'single-split-train-data'
fi

if [ ! -d "./single-split-test-data" ];
    then
    echo 'folder single-split-test-data does not exist!'
    mkdir single-split-test-data
fi

# Simple script for converting test data for the Fully Connected NN training/test

python converter.py -s 0.1 -d 1 > /dev/null 2>&1 &
python converter.py -s 0.2 -d 1 > /dev/null 2>&1 &
python converter.py -s 0.3 -d 1 > /dev/null 2>&1 &
python converter.py -s 0.4 -d 1 > /dev/null 2>&1 &
