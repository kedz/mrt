python dataset_preprocessing/preprocess_viggo.py \
    data/orig/viggo/viggo-train.csv \
    data/orig/viggo/viggo-valid.csv \
    data/orig/viggo/viggo-test.csv \
    data/viggo/

python dataset_preprocessing/add_simple_orders.py \
    data/viggo/viggo.train.jsonl \
    data/viggo/viggo.valid.jsonl \
    data/viggo/viggo.test.jsonl 
