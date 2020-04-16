#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$(readlink -f $SCRIPT_DIR/../data)

E2E_ORIG=$DATA_DIR/orig/E2E
E2E_OUTPUT=$DATA_DIR/E2E
E2E_TRAIN_ORIG=$E2E_ORIG/train-fixed.no-ol.csv
E2E_TRAIN=$E2E_OUTPUT/E2E.train.jsonl
E2E_VALID_ORIG=$E2E_ORIG/devel-fixed.no-ol.csv
E2E_VALID=$E2E_OUTPUT/E2E.valid.jsonl
E2E_TEST_ORIG=$E2E_ORIG/test-fixed.csv

#?echo "Formatting E2E training dataset..."
#?$SCRIPT_DIR/format_data.py E2E \
#?    $E2E_ORIG/train-fixed.no-ol.csv \
#?    $E2E_OUTPUT/E2E.train.jsonl \
#?    --procs 12
#?
#?echo "Formatting E2E validation dataset..."
#?$SCRIPT_DIR/format_data.py E2E \
#?    $E2E_ORIG/devel-fixed.no-ol.csv \
#?    $E2E_OUTPUT/E2E.valid.jsonl \
#?    --procs 12
#?
#echo "Formatting E2E test dataset..."
#$SCRIPT_DIR/format_data.py E2E \
#    $E2E_ORIG/test-fixed.csv \
#    $E2E_OUTPUT/E2E.test.jsonl \
#    --procs 12 \
#    --test
#
#exit



#echo "Formatting Viggo dataset..."
VIGGO_ORIG=$DATA_DIR/orig/Viggo
VIGGO_OUTPUT=$DATA_DIR/Viggo
#$SCRIPT_DIR/format_data.py Viggo \
#    $VIGGO_ORIG/viggo-train.csv \
#    $VIGGO_OUTPUT/Viggo.train.jsonl
#
#echo "Formatting Viggo dataset..."
#VIGGO_ORIG=$DATA_DIR/orig/Viggo
#VIGGO_OUTPUT=$DATA_DIR/Viggo
#$SCRIPT_DIR/format_data.py Viggo \
#    $VIGGO_ORIG/viggo-valid.csv \
#    $VIGGO_OUTPUT/Viggo.valid.jsonl
#
#echo "Formatting Viggo dataset..."
#VIGGO_ORIG=$DATA_DIR/orig/Viggo
#VIGGO_OUTPUT=$DATA_DIR/Viggo
#$SCRIPT_DIR/format_data.py Viggo \
#    $VIGGO_ORIG/viggo-test.csv \
#    $VIGGO_OUTPUT/Viggo.test.jsonl \
#    --test
#
#


$SCRIPT_DIR/make_slot_filler_frequencies.py \
    E2E \
    $E2E_OUTPUT/E2E.train.jsonl \
    $E2E_OUTPUT/E2E.slot_filler.freqs.json

$SCRIPT_DIR/add_simple_orders2.py \
    E2E \
    $E2E_OUTPUT/E2E.slot_filler.freqs.json \
    $E2E_OUTPUT/E2E.train.jsonl \
    $E2E_OUTPUT/E2E.valid.jsonl \
    $E2E_OUTPUT/E2E.test.jsonl 

exit


$SCRIPT_DIR/make_slot_filler_frequencies.py \
    Viggo \
    $VIGGO_OUTPUT/Viggo.train.jsonl \
    $VIGGO_OUTPUT/Viggo.slot_filler.freqs.json

$SCRIPT_DIR/add_simple_orders2.py \
    Viggo \
    $VIGGO_OUTPUT/Viggo.slot_filler.freqs.json \
    $VIGGO_OUTPUT/Viggo.train.jsonl \
    $VIGGO_OUTPUT/Viggo.valid.jsonl \
    $VIGGO_OUTPUT/Viggo.test.jsonl \



exit


echo "Formatting E2E dataset..."
E2E_ORIG=$DATA_DIR/orig/E2E
E2E_OUTPUT=$DATA_DIR/E2E
$SCRIPT_DIR/format_data.py E2E \
    $E2E_ORIG/train-fixed.no-ol.csv \
    $E2E_OUTPUT/E2E.train.jsonl --procs 12

exit


#    $VIGGO_ORIG/viggo-valid.csv \
#    $VIGGO_ORIG/viggo-test.csv \
#




#echo "Formatting E2E dataset..."
#E2E_ORIG=$DATA_DIR/orig/E2E
#E2E_OUTPUT=$DATA_DIR/E2E
#$SCRIPT_DIR/format_E2E.py E2E \
#    $E2E_ORIG/train-fixed.no-ol.csv \
#    $E2E_ORIG/devel-fixed.no-ol.csv \
#    $E2E_ORIG/test-fixed.csv \
#    $E2E_OUTPUT
#
#
##echo "Formatting Viggo dataset..."
##VIGGO_ORIG=$DATA_DIR/orig/Viggo
##VIGGO_OUTPUT=$DATA_DIR/Viggo
##$SCRIPT_DIR/format_E2E.py Viggo \
##    $VIGGO_ORIG/viggo-train.csv \
##    $VIGGO_ORIG/viggo-valid.csv \
##    $VIGGO_ORIG/viggo-test.csv \
##    $VIGGO_OUTPUT
