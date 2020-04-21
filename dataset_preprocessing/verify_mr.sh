#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$(readlink -f $SCRIPT_DIR/../data)

$SCRIPT_DIR/verify_mr.py Viggo \
    $SCRIPT_DIR/Viggo.train.verified.final.jsonl \
    $SCRIPT_DIR/Viggo.train.verified.final.jsonl \
    $DATA_DIR/orig/Viggo/viggo-train.csv

$SCRIPT_DIR/verify_mr.py Viggo \
    $SCRIPT_DIR/Viggo.valid.verified.final.jsonl \
    $SCRIPT_DIR/Viggo.valid.verified.final.jsonl \
    $DATA_DIR/orig/Viggo/viggo-valid.csv

$SCRIPT_DIR/verify_mr.py E2E \
    $SCRIPT_DIR/E2E.train.verified.final.jsonl \
    $SCRIPT_DIR/E2E.train.verified.final.jsonl \
    $DATA_DIR/orig/E2E/train-fixed.no-ol.csv

$SCRIPT_DIR/verify_mr.py E2E \
    $SCRIPT_DIR/E2E.valid.verified.final.jsonl \
    $SCRIPT_DIR/E2E.valid.verified.final.jsonl \
    $DATA_DIR/orig/E2E/devel-fixed.no-ol.csv
