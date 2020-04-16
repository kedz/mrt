#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR=$(readlink -f $SCRIPT_DIR/../data)

ORIG_VIGGO_DATA_DIR=$DATA_DIR/orig/Viggo/
mkdir -p $ORIG_VIGGO_DATA_DIR
wget -O $ORIG_VIGGO_DATA_DIR/viggo-v1.zip http://nldslab.soe.ucsc.edu/viggo/viggo-v1.zip
unzip $ORIG_VIGGO_DATA_DIR/viggo-v1.zip -d $ORIG_VIGGO_DATA_DIR
rm $ORIG_VIGGO_DATA_DIR/viggo-v1.zip

ORIG_E2E_DATA_DIR=$DATA_DIR/orig/E2E/
mkdir -p $ORIG_E2E_DATA_DIR

wget -O $ORIG_E2E_DATA_DIR/train-fixed.no-ol.csv https://github.com/tuetschek/e2e-cleaning/raw/master/cleaned-data/train-fixed.no-ol.csv
wget -O $ORIG_E2E_DATA_DIR/devel-fixed.no-ol.csv https://github.com/tuetschek/e2e-cleaning/raw/master/cleaned-data/devel-fixed.no-ol.csv
wget -O $ORIG_E2E_DATA_DIR/test-fixed.csv https://github.com/tuetschek/e2e-cleaning/raw/master/cleaned-data/test-fixed.csv

echo "Viggo dataset downloaded to: $ORIG_VIGGO_DATA_DIR"
echo "E2E dataset downloaded to: $ORIG_E2E_DATA_DIR"
