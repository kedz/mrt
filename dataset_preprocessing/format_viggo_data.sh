#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$(readlink -f $SCRIPT_DIR/../data)

VIGGO_ORIG=$DATA_DIR/orig/Viggo
VIGGO_TRAIN_ORIG=$VIGGO_ORIG/viggo-train.csv
VIGGO_VALID_ORIG=$VIGGO_ORIG/viggo-valid.csv
VIGGO_TEST_ORIG=$VIGGO_ORIG/viggo-test.csv

VIGGO_OUTPUT=$DATA_DIR/Viggo
VIGGO_TRAIN=$VIGGO_OUTPUT/Viggo.train.jsonl
VIGGO_VALID=$VIGGO_OUTPUT/Viggo.valid.jsonl
VIGGO_VALID_AGG=$VIGGO_OUTPUT/Viggo.valid.agg.jsonl
VIGGO_TEST=$VIGGO_OUTPUT/Viggo.test.jsonl
VIGGO_TEST_AGG=$VIGGO_OUTPUT/Viggo.test.agg.jsonl
VIGGO_FREQ=$VIGGO_OUTPUT/Viggo.slot_filler.freqs.json
VIGGO_PHRASES=$VIGGO_OUTPUT/Viggo.phrases.jsonl
VIGGO_TEMPLATES=$VIGGO_OUTPUT/Viggo.templates.jsonl
VIGGO_PHRASES_NOOL=$VIGGO_OUTPUT/Viggo.phrases.no-ol.jsonl
VIGGO_TEMPLATES_NOOL=$VIGGO_OUTPUT/Viggo.templates.no-ol.jsonl


echo "Formatting Viggo training data..."
$SCRIPT_DIR/format_data.py Viggo \
    $VIGGO_TRAIN_ORIG \
    $VIGGO_TRAIN

echo "Formatting Viggo validation data..."
$SCRIPT_DIR/format_data.py Viggo \
    $VIGGO_VALID_ORIG \
    $VIGGO_VALID

echo "Formatting Viggo testing data..."
$SCRIPT_DIR/format_data.py Viggo \
    $VIGGO_TEST_ORIG \
    $VIGGO_TEST \
    --test \

echo "Making Viggo slot filler count data..."
$SCRIPT_DIR/make_slot_filler_frequencies.py \
    Viggo \
    $VIGGO_TRAIN \
    $VIGGO_FREQ

echo "Adding Viggo data derived orders..."
$SCRIPT_DIR/add_simple_orders.py \
    Viggo \
    $VIGGO_FREQ \
    $VIGGO_TRAIN \
    $VIGGO_VALID \
    $VIGGO_TEST

echo "Aggregating validation dataset."
$SCRIPT_DIR/aggregate_dataset.py \
    Viggo \
    $VIGGO_VALID \
    $VIGGO_VALID_AGG 
    
echo "Aggregating test dataset."
$SCRIPT_DIR/aggregate_dataset.py \
    Viggo \
    $VIGGO_TEST \
    $VIGGO_TEST_AGG \
    --test 

echo "Generate Viggo Phrase Data"
$SCRIPT_DIR/generate_phrases.py \
    Viggo \
    $VIGGO_FREQ \
    $VIGGO_TRAIN \
    $VIGGO_PHRASES \
    --procs 16

echo "Generate Viggo Templates Data"
$SCRIPT_DIR/generate_templates.py \
    Viggo \
    $VIGGO_FREQ \
    $VIGGO_TEMPLATES 

echo "Remove test set overlap..."
$SCRIPT_DIR/remove_overlap.py \
    Viggo \
    $VIGGO_PHRASES \
    $VIGGO_PHRASES_NOOL \
    $VIGGO_TEMPLATES \
    $VIGGO_TEMPLATES_NOOL \
    $VIGGO_TEST
