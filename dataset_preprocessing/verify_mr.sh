#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$(readlink -f $SCRIPT_DIR/../data)

./verify_mr.py Viggo \
    Viggo.train.verified.final.jsonl \
    Viggo.train.verified.final.jsonl \
    $DATA_DIR/orig/Viggo/viggo-train.csv

./verify_mr.py Viggo \
    Viggo.valid.verified.final.jsonl \
    Viggo.valid.verified.final.jsonl \
    $DATA_DIR/orig/Viggo/viggo-valid.csv
