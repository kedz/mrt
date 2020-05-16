import os
from pathlib import Path
from plum2 import seeds, work_dir, HP
from mrt.utils import (
    setup_vocab, setup_training_data, setup_validation_data,
    make_batches, setup_model, setup_trainer, setup_generator
)

seeds(234222452)

assert os.getenv("MRT_DATASET") is not None
assert os.getenv("MRT_LIN_STRAT") is not None
assert os.getenv("MRT_DELEX") is not None

dataset = os.getenv("MRT_DATASET")
lin_strat = os.getenv("MRT_LIN_STRAT")
is_delex = os.getenv("MRT_DELEX") == "delex"

work_dir(f"experiments/{dataset}")

vcb_lin_strat = "rule" if lin_strat in ['nlm', 'bglm'] else lin_strat
mr_vcb, utt_vcb = setup_vocab(dataset, vcb_lin_strat, is_delex)

with plum2.dataset('valid-agg') as va_agg_ds:
    part = 'valid.no-ol.agg.jsonl' if dataset == 'E2E' else 'valid.agg.jsonl'
    va_agg_ds.jsonl(f"data/{dataset}/{dataset}.{part}")
 
with plum2.dataset('test-agg') as te_agg_ds:
    part = 'test.agg.jsonl'
    te_agg_ds.jsonl(f"data/{dataset}/{dataset}.{part}")
 
MOD = HP('MOD', '', description='model path')
INP = HP("INP", '', description='input path')
RS = HP("RS", '', description='rule set')
ORD = HP('ORD', 'rule_delex', description='generator order')
ORIG_MR = HP('ORIG_MR', 'false', description='use orig mr')
OUT = HP("OUT", '', description='output path')

#PRE = HP('PRE', '', description='output prefix')

from plum2.tasks import TASKS
from mrt.generate_text import GenerateText
TASKS['generate'] = GenerateText(MOD, RS, INP, ORD, mr_vcb, utt_vcb, OUT, orig_mr=ORIG_MR)

#setup_generator(MOD, va_agg_ds, mr_vcb, utt_vcb, ORD, PRE, 'generate-valid') 
#setup_generator(MOD, te_agg_ds, mr_vcb, utt_vcb, ORD, PRE, 'generate-test') 



