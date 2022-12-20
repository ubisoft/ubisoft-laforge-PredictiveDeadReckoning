#!/bin/bash

# Simple script for processing test data into a digestable form for the LSTM NN

python converter.py -s 0.1  > /dev/null 2>&1 &
python converter.py -s 0.2  > /dev/null 2>&1 &
python converter.py -s 0.3  > /dev/null 2>&1 &
python converter.py -s 0.4  > /dev/null 2>&1 &
