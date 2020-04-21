#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$(readlink -f $SCRIPT_DIR/../data)

E2E_ORIG=$DATA_DIR/orig/E2E
E2E_TRAIN_ORIG=$E2E_ORIG/train-fixed.no-ol.csv
E2E_VALID_ORIG=$E2E_ORIG/devel-fixed.no-ol.csv
E2E_TEST_ORIG=$E2E_ORIG/test-fixed.csv

E2E_OUTPUT=$DATA_DIR/E2E
E2E_TRAIN=$E2E_OUTPUT/E2E.train.jsonl
E2E_VALID=$E2E_OUTPUT/E2E.valid.jsonl
E2E_TRAIN_NOOL=$E2E_OUTPUT/E2E.train.no-ol.jsonl
E2E_VALID_NOOL=$E2E_OUTPUT/E2E.valid.no-ol.jsonl
E2E_VALID_NOOL_AGG=$E2E_OUTPUT/E2E.valid.no-ol.agg.jsonl
E2E_TEST=$E2E_OUTPUT/E2E.test.jsonl
E2E_TEST_AGG=$E2E_OUTPUT/E2E.test.agg.jsonl
E2E_FREQ=$E2E_OUTPUT/E2E.slot_filler.freqs.json
E2E_PHRASES=$E2E_OUTPUT/E2E.phrases.jsonl
E2E_TEMPLATES=$E2E_OUTPUT/E2E.templates.jsonl
E2E_PHRASES_NOOL=$E2E_OUTPUT/E2E.phrases.no-ol.jsonl
E2E_TEMPLATES_NOOL=$E2E_OUTPUT/E2E.templates.no-ol.jsonl

echo "Formatting E2E training dataset..."
$SCRIPT_DIR/format_data.py E2E \
    $E2E_TRAIN_ORIG \
    $E2E_TRAIN \
    --procs 12

echo "Formatting E2E validation dataset..."
$SCRIPT_DIR/format_data.py E2E \
    $E2E_VALID_ORIG \
    $E2E_VALID \
    --procs 12

echo "Formatting E2E test dataset..."
$SCRIPT_DIR/format_data.py E2E \
    $E2E_TEST_ORIG \
    $E2E_TEST \
    --procs 12 \
    --test

echo "Making E2E slot filler count data..."
$SCRIPT_DIR/make_slot_filler_frequencies.py \
    E2E \
    $E2E_TRAIN \
    $E2E_FREQ

echo "Adding E2E data derived orders..."
$SCRIPT_DIR/add_simple_orders.py \
    E2E \
    $E2E_FREQ \
    $E2E_TRAIN \
    $E2E_VALID \
    $E2E_TEST
    
echo "Remove test set overlap..."
$SCRIPT_DIR/remove_overlap.py \
    E2E \
    $E2E_TRAIN \
    $E2E_TRAIN_NOOL \
    $E2E_VALID \
    $E2E_VALID_NOOL \
    $E2E_TEST
 
echo "Aggregating validation dataset."
$SCRIPT_DIR/aggregate_dataset.py \
    E2E \
    $E2E_VALID_NOOL \
    $E2E_VALID_NOOL_AGG
    
echo "Aggregating test dataset."
$SCRIPT_DIR/aggregate_dataset.py \
    E2E \
    $E2E_TEST \
    $E2E_TEST_AGG \
    --test 

echo "Generate E2E Phrase Data"
$SCRIPT_DIR/generate_phrases.py \
    E2E \
    $E2E_FREQ \
    $E2E_TRAIN \
    $E2E_PHRASES \
    --procs 16

echo "Generate E2E Templates Data"
$SCRIPT_DIR/generate_templates.py \
    E2E \
    $E2E_FREQ \
    $E2E_TEMPLATES 

echo "Remove test set overlap..."
$SCRIPT_DIR/remove_overlap.py \
    E2E \
    $E2E_PHRASES \
    $E2E_PHRASES_NOOL \
    $E2E_TEMPLATES \
    $E2E_TEMPLATES_NOOL \
    $E2E_TEST
