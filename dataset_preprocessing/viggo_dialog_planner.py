from pathlib import Path

from plum2 import seeds, work_dir, HP
from plum2.optimizer import Adam

from mrt.dialog_planner.model import dialog_planner
from mrt.dialog_planner.pointer_cross_entropy import PointerCrossEntropy
from mrt.dialog_planner.dialog_metric import DialogMetric
#from mrt.dialog_planner.generate_dialog_plans import add_dialog_plans
#from mrt.dialog_planner.eval_dialog_plans import eval_dialog_plans


SOS = '<sos>'
EOS = '<eos>'

def make_encoder_input(example):
    tokens = example['source']['sequence']['random']
    # remove duplicate tokens 
    tokens = [tok for i, tok in enumerate(tokens) if tok not in tokens[:i]]
    return tokens + [EOS]

def make_decoder_input(example):
    tokens = example['source']['sequence']['oracle'][2:]
    # remove duplicate tokens 
    tokens = [tok for i, tok in enumerate(tokens) if tok not in tokens[:i]]
    return [SOS] + tokens

def make_decoder_output(example):
    ei_tokens = make_encoder_input(example)
    di_tokens = make_decoder_input(example)
    
    do_indices = []
    for tok in di_tokens[1:] + [EOS]:
        idx = ei_tokens.index(tok)
        do_indices.append(idx)
     
    return do_indices

seeds(234222452)
save_dir = Path("experiments", "viggo", "dialog_planner")
work_dir(save_dir)

LR = HP('LR', 0.0001, description='learning rate')
BS = HP('BS', 128, description='batch size')
VBS = HP(
    'VBS', 32, 
    description='validation set batch size (smaller for beam search)')
L = HP('L', 2, description='num layers')

with plum2.dataset("viggo.train") as tr:
    tr.jsonl("data/viggo/viggo.train.jsonl")

    with tr.vocab("mr") as mrs:
        mrs.apply_func(make_encoder_input).pad().start(SOS).stop(EOS)

    with tr.feature('encoder_input') as x:
        x.apply_func(make_encoder_input).lookup(mrs)\
            .batch_varlen_seq(mrs.pad_index)

    with tr.feature('decoder_input') as y_in:
        y_in.apply_func(make_decoder_input).lookup(mrs)\
            .batch_varlen_seq(mrs.pad_index)

    with tr.feature('target') as y_out:
        y_out.apply_func(make_decoder_output)\
            .batch_varlen_seq(-1)

with plum2.dataset("viggo.valid") as va:
    va.jsonl("data/viggo/viggo.valid.jsonl")

    with va.feature('encoder_input') as x:
        x.apply_func(make_encoder_input).lookup(mrs)\
            .batch_varlen_seq(mrs.pad_index)

    with va.feature('decoder_input') as y_in:
        y_in.apply_func(make_decoder_input).lookup(mrs)\
            .batch_varlen_seq(mrs.pad_index)

    with va.feature('target') as y_out:
        y_out.apply_func(make_decoder_output)\
            .batch_varlen_seq(-1)

    with va.feature('pretty_target') as y:
        y.apply_func(make_decoder_input).apply_func(lambda x: x[1:])

    with va.feature('da') as da:
        da.apply_indexers(['source', 'mr', 'da'])

tr_batches = tr.batch_iter(BS, workers=4)
va_batches = va.batch_iter(VBS, workers=4)
   
with dialog_planner('tf-pointer') as m:
    m.emb(512).hidden(2048).layers(L).dropout(0.1).vocab(mrs).beam_search()

with dialog_planner('tf-pointer-tied') as m_tied:
    m_tied.emb(512).hidden(2048).layers(L).dropout(0.1).vocab(mrs)\
        .beam_search().tie_embs()

def make_prefix(suffix):
    def prefix():
        return f"adam_lr={float(LR)}_layers={int(L)}{suffix}"
    return prefix

for model, suffix in [[m, ''], [m_tied, '_tied']]:
    

    opt = Adam(LR)
    with plum2.trainer(f"train{suffix}") as trainer:
        trainer.model(model).optim(opt)\
            .train(tr_batches).valid(va_batches).epochs(50)\
            .loss(PointerCrossEntropy(padding_index=-1, label_smoothing=0.1))\
            .valid_metrics(
                DialogMetric(
                    lambda m, b, fs: [s[:-1] 
                                      for s in fs['search']['beam'].output()],
                    lambda m, b, fs: b['pretty_target'],
                    lambda m, b, fs: b['da'],
                )
            ).save_prefix(make_prefix(suffix))\
            .save_best()

        model_path = (save_dir / f'trainer{suffix}'/ 'model_checkpoints' 
            / 'optimal.pkl')
#
#        add_dialog_plans('generate_valid', model_path, 'data/viggo/viggo.valid.jsonl',
#                 tr.pipelines, mrs)
#
#add_dialog_plans('generate_test', model_path, 'data/viggo/viggo.test.jsonl',
#                 tr.pipelines, mrs)
#
##eval_dialog_plans('eval_valid', model_path, va_batches, mrs)
#eval_dialog_plans('eval_valid', model_path, va_batches, mrs, beam=4)
#
##eval_dialog_plans(
##    'eval_test',
##    model_path,
##    tr_batches,
##    mrs
##)
