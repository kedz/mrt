import os
from pathlib import Path
from plum2 import seeds, work_dir, HP

import mrt.viggo.mr_utils
import mrt.e2e.mr_utils
from mrt.utils import (
    setup_vocab, setup_training_data, setup_validation_data,
    make_batches, setup_model, setup_trainer, setup_generator
)
            

seeds(234222452)

assert os.getenv("MRT_DATASET") is not None
assert os.getenv("MRT_LIN_STRAT") is not None
assert os.getenv("MRT_DELEX") is not None
assert os.getenv("MRT_EVAL_SCRIPT") is not None

dataset = os.getenv("MRT_DATASET")
lin_strat = os.getenv("MRT_LIN_STRAT")
is_delex = os.getenv("MRT_DELEX") == "delex"
rnn_attn = os.getenv("MRT_RNN_ATTN", 'luong-general')
rnn_dir = os.getenv("MRT_RNN_DIR", 'bi')
optimizer = os.getenv("MRT_OPT", 'sgd')
model_type = os.getenv("MRT_MODEL_TYPE", "gru")
tie_dec_emb = os.getenv("MRT_TIE_EMB", "true") == "true"
eval_script = os.getenv("MRT_EVAL_SCRIPT")

src_seq = f"{lin_strat}_{'delex' if is_delex else 'lex'}" 

work_dir(f"experiments/{dataset}")

WKRS = HP('WKRS', 2, description='data loader processes')
T = HP('T', 500, description='max training epochs')
LR = HP('LR', 0.1, description='learning rate')
WD = HP('WD', 0.0, description='weight decay')
LS = HP('LS', 0.1, description='label smoothing')
L = HP('L', 2, description='num layers')

mr_vcb, utt_vcb = setup_vocab(dataset, lin_strat, is_delex)

tr_ds = setup_training_data(dataset, lin_strat, is_delex, mr_vcb, utt_vcb)
tr_phrase_ds = setup_training_data(dataset, lin_strat, is_delex, 
                                   mr_vcb, utt_vcb, include_phrases=True)
tr_templ_ds = setup_training_data(dataset, lin_strat, is_delex, 
                                  mr_vcb, utt_vcb, include_templates=True)
tr_phrase_templ_ds = setup_training_data(dataset, lin_strat, is_delex, 
                                         mr_vcb, utt_vcb, include_phrases=True,
                                         include_templates=True)

va_ds = setup_validation_data(dataset, lin_strat, is_delex, mr_vcb, utt_vcb)

tr_batches = make_batches(tr_ds, 128, WKRS, lin_strat, is_delex)
tr_phrase_batches = make_batches(tr_phrase_ds, 128, WKRS, lin_strat, is_delex)
tr_template_batches = make_batches(tr_templ_ds, 128, WKRS, lin_strat, is_delex)
tr_phrase_template_batches = make_batches(tr_phrase_templ_ds, 128, WKRS, 
                                          lin_strat, is_delex)

va_batches = make_batches(va_ds, 1 if model_type == 'transformer' else 64, 
                          WKRS, lin_strat, is_delex)

model = setup_model(model_type, rnn_dir, L, rnn_attn, tie_dec_emb,
                    mr_vcb, utt_vcb, beam_size=8)

if dataset == 'Viggo':
    mr_utils = mrt.viggo.mr_utils
elif dataset == 'E2E':
    mr_utils = mrt.e2e.mr_utils
else:
    raise Exception(f"Bad dataset: {dataset}")

def make_template():
    if model_type in ['gru', 'lstm']:
        TEMPLATE = "rnn/{lin_strat}_{delex}/{arch}_{dir}_L={layers}_attn={attn}_tied={tied}_{opt}_lr={lr}_wd={wd}_ls={ls}"
        return TEMPLATE.format(
            lin_strat=lin_strat, delex='delex' if is_delex else 'lex', 
            dir=rnn_dir,
            arch=model_type, layers=int(L), attn=rnn_attn, tied=tie_dec_emb,
            opt=optimizer,
            lr=float(LR), wd=float(WD), ls=float(LS))
    else:
        TEMPLATE = "transformer/{lin_strat}_{delex}/posemb_L={layers}_tied={tied}_{opt}_lr={lr}_wd={wd}_ls={ls}"
        return TEMPLATE.format(
            lin_strat=lin_strat, delex='delex' if is_delex else 'lex', 
            arch=model_type, layers=int(L), tied=tie_dec_emb,
            opt=optimizer,
            lr=float(LR), wd=float(WD), ls=float(LS))

setup_trainer(model, optimizer, LR, WD, LS, tr_batches, va_batches, T, 
              utt_vcb, eval_script, mr_utils, make_template, "base")

setup_trainer(model, optimizer, LR, WD, LS, tr_phrase_batches, va_batches, T, 
              utt_vcb, eval_script, mr_utils, make_template, "base+phrases")

setup_trainer(model, optimizer, LR, WD, LS, tr_template_batches, va_batches, T, 
              utt_vcb, eval_script, mr_utils, make_template, "base+templates")

setup_trainer(model, optimizer, LR, WD, LS, 
              tr_phrase_template_batches, va_batches,
              T, utt_vcb, eval_script, mr_utils, make_template, 
              "base+phrases+templates")

model_paths = {
    ("Viggo", "transformer", "rule", True): (
        "posemb_L=2_tied=False_adamtri_lr=0.1_wd=0.0_ls=0.1"),
    ("Viggo", "transformer", "inc_freq", True): (
        "posemb_L=2_tied=False_adamtri_lr=0.1_wd=0.0_ls=0.1"),
    ("Viggo", "transformer", "inc_freq_fixed", True): (
        "posemb_L=2_tied=False_adamtri_lr=0.1_wd=0.0_ls=0.1"),

    ("Viggo", "gru", "bi", "rule", True): (
        "gru_bi_L=2_attn=bahdanau_tied=False_adam_lr=1e-05_wd=0.0_ls=0.1/"),
    ("Viggo", "gru", "bi", "inc_freq_fixed", True): ("foo"),


    ("Viggo", "lstm", "bi", "inc_freq_fixed", True): (
        "foo"),
    ("Viggo", "lstm", "bi", "random", True): (
        "bar"),

    
    ("E2E", "gru", "bi", "rule", False): ('foo'),
    ("E2E", "gru", "bi", "inc_freq_fixed", False): ('foo'),

}



with plum2.dataset('valid-agg') as va_agg_ds:
    part = 'valid.no-ol.agg.jsonl' if dataset == 'E2E' else 'valid.agg.jsonl'
    va_agg_ds.jsonl(f"data/{dataset}/{dataset}.{part}")
 

GENORDER = HP('ORD', 'rule_delex', description='generator order')

for ver in ['base', 'base+phrases', 'base+templates', 
            'base+phrases+templates']:
    if model_type == 'transformer':
        output_prefix = (
            model_type + "." + lin_strat + "_" 
            + ('delex' if is_delex else 'lex')
            + ".to" 
        )
        key = (dataset, model_type, lin_strat, is_delex)
        model_path = (
            Path('experiments', dataset, f'train_{ver}', "transformer",
                          lin_strat + '_delex' if is_delex else '_lex') /
                model_paths[key] / "model_checkpoints" / "optimal.pkl")

    else:
        output_prefix = (
            model_type + "." + rnn_dir + "." + lin_strat + "_" 
            + ('delex' if is_delex else 'lex')
            + ".to" 
        )
        key = (dataset, model_type, rnn_dir, lin_strat, is_delex)
        model_path = (
            Path('experiments', dataset, f'train_{ver}', "rnn",
                          lin_strat + '_delex' if is_delex else '_lex') /
                model_paths[key] / "model_checkpoints" / "optimal.pkl")
    setup_generator(model_path, va_agg_ds, mr_vcb, utt_vcb, GENORDER,
                    output_prefix,
                    f'generate_valid[{ver}]') 


