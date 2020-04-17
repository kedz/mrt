from plum2 import seeds, work_dir, HP
from plum2.loss import ClassCrossEntropy
from plum2.logger import ModelOutputLogger
from plum2.metrics import Script
from plum2.optimizer import SGD, Adam

import mrt.viggo.mr_utils
import mrt.e2e.mr_utils


DS = 'Viggo'
LEX = 'delex'
SEQ = 'inc_freq_delex'

if DS == 'Viggo':
    mr_utils = mrt.viggo.mr_utils
elif DS == 'E2E':
    mr_utils = mrt.e2e.mr_utils
else:
    raise Exception(f"Bad DS: {DS}")

seeds(234222452)
work_dir(f"experiments/hp_search/{DS}/{SEQ}/rnn/")

LR = HP('LR', 0.1, description='learning rate')
WD = HP('WD', 1e-5, description='weight decay')
LS = HP('LS', 0.1, description='label smoothing')
L = HP('L', 2, description='num layers')
C = HP('C', 'gru', description='rnn cell')

with plum2.dataset(f"{DS}.train") as tr_ds:
    tr_ds.jsonl(f"data/{DS}/{DS}.train.jsonl")

    with tr_ds.vocab("mr") as mr_vcb:
        mr_vcb.apply_indexers(['source', 'sequence', SEQ])\
            .pad().start().stop()

    with tr_ds.vocab("utt") as utt_vcb:
        utt_vcb.apply_indexers(['target', "sequence", LEX]).pad().start().stop()\
            .unk().at_least(3)
        
    with tr_ds.feature("encoder_input") as x:
        x.apply_indexers(['source', 'sequence', SEQ])\
            .lookup(mr_vcb)\
            .prepend(mr_vcb.start_index).extend(mr_vcb.stop_index)\
            .batch_varlen_seq(mr_vcb.pad_index)

    with tr_ds.feature("decoder_input") as y:
        y.apply_indexers(['target', "sequence", LEX]).lookup(utt_vcb)\
            .prepend(utt_vcb.start_index).batch_varlen_seq(utt_vcb.pad_index)

    with tr_ds.feature("target") as y:
        y.apply_indexers(["target", "sequence", LEX]).lookup(utt_vcb)\
            .extend(utt_vcb.stop_index).batch_varlen_seq(utt_vcb.pad_index)
                    
with plum2.dataset(f"{DS}.valid") as va_ds:

    va_ds.jsonl(f"data/{DS}/{DS}.valid.jsonl")

    with va_ds.feature("encoder_input") as x:
        x.apply_indexers(['source', 'sequence', SEQ])\
            .lookup(mr_vcb)\
            .prepend(mr_vcb.start_index).extend(mr_vcb.stop_index)\
            .batch_varlen_seq(mr_vcb.pad_index)
    
    with va_ds.feature("mr") as x:
        x.apply_indexers(['source', 'mr'])
    with va_ds.feature("encoder_input_pretty") as x:
        x.apply_indexers(['source', 'sequence', SEQ])
    with va_ds.feature("decoder_input") as y:
        y.apply_indexers(["target", "sequence", LEX]).lookup(utt_vcb)\
            .prepend(utt_vcb.start_index).batch_varlen_seq(utt_vcb.pad_index)
    with va_ds.feature("target") as y:
        y.apply_indexers(["target", "sequence", LEX]).lookup(utt_vcb)\
            .extend(utt_vcb.stop_index).batch_varlen_seq(utt_vcb.pad_index)
    with va_ds.feature("target_pretty") as y:
        y.apply_indexers(["target", "reference"])

tr_batches = tr_ds.batch_iter(128, sort=True, 
    sort_key=lambda x: len(x['source']['sequence'][SEQ]), workers=4)
va_batches = va_ds.batch_iter(128, sort=True, 
    sort_key=lambda x: len(x['source']['sequence'][SEQ]), workers=4)

def update_lr(epoch, trainer, results):
    curr_result = results[-1]['valid']['loss']['detail']['cross_entropy']
    improved = any([
        curr_result < x['valid']['loss']['detail']['cross_entropy']
        for x in results[-5-1:-1]
    ])
    if epoch > 5 and not improved:
        for p in trainer.optimizer.impl.param_groups:
            lr = p['lr']
            lr = lr * .99
            print(f"new LR={lr}")
            p['lr'] = lr

def target_pretty(m, b, fs): 
    return [f"{x}\n" for x in b["target_pretty"]]

def beam_out(m, b, fs): 
    outputs = [] 
    for mr, tokens in zip(b['mr'], fs["search"]["beam"].output()):
        text = mr_utils.detokenize(tokens)
        if LEX == 'delex':
            text = mr_utils.lexicalize_string(text, **mr['slots'])
        outputs.append(text)
    return outputs

def make_templ(attn, opt, bidir):
    def new_tmpl():
        return f"{attn}_bidir_{bidir}_{opt}_lr={float(LR)}_wd={float(WD)}_ls_{float(LS)}_l_{int(L)}_cell_{str(C)}"
    return new_tmpl

for attn in plum2.layer.attention.TYPES:

    opt = SGD(LR, weight_decay=WD)
    with plum2.seq2seq("rnn-uni") as m_uni:
        m_uni.arch(C).emb(512).hidden(512).layers(L).bi(False)\
            .dropout(0.1).attn(attn)\
            .pass_thru_bridge()\
            .enc_vocab(mr_vcb).dec_vocab(utt_vcb)\
            .beam_search(max_steps=100, beam_size=4)

    with plum2.trainer(f"hps_rnn-uni_sgd_{attn}") as trainer:
        trainer.model(m_uni).optim(opt)\
            .train(tr_batches).valid(va_batches).epochs(500)\
            .loss(ClassCrossEntropy(padding_index=utt_vcb.pad_index, 
                                    label_smoothing=LS))\
            .add_callback(update_lr)\
            .valid_metrics(
                Script(
                    "../../v2/eval/e2e-metrics/measure_scores.py",
                    target_pretty,
                    beam_out,
                    lambda x: {
                        l.split(": ")[0]: float(l.split(": ")[1]) 
                        for l in x.decode("utf8").strip().split("\n")[-5:]
                    }, lambda x: x["BLEU"], False,
                    every=10
                )
            )\
            .save_prefix(make_templ(attn, 'sgd', False))

    if attn != 'luong-dot':    
        opt = SGD(LR, weight_decay=WD)
        with plum2.seq2seq("rnn-bi") as m_bi:
            m_bi.arch(C).emb(512).hidden(512).layers(L).bi(True)\
                .dropout(0.1).attn(attn)\
                .enc_vocab(mr_vcb).dec_vocab(utt_vcb)\
                .beam_search(max_steps=100, beam_size=4)

        with plum2.trainer(f"hps_rnn-bi_sgd_{attn}") as trainer:
            trainer.model(m_bi).optim(opt)\
                .train(tr_batches).valid(va_batches).epochs(500)\
                .loss(ClassCrossEntropy(padding_index=utt_vcb.pad_index, 
                                        label_smoothing=LS))\
                .add_callback(update_lr)\
                .valid_metrics(
                    Script(
                        "../../v2/eval/e2e-metrics/measure_scores.py",
                        target_pretty,
                        beam_out,
                        lambda x: {
                            l.split(": ")[0]: float(l.split(": ")[1]) 
                            for l in x.decode("utf8").strip().split("\n")[-5:]
                        }, lambda x: x["BLEU"], False,
                        every=10
                    )
                )\
                .save_prefix(make_templ(attn, 'sgd', False))






#        .save_best()
#
#opt = Adam(LR, weight_decay=WD)
#with plum2.trainer(f"train_gru_adam") as trainer:
#    trainer.model(m).optim(opt)\
#        .train(tr_batches).valid(va_batches).epochs(500)\
#        .loss(ClassCrossEntropy(padding_index=utt_vcb.pad_index, 
#                                label_smoothing=LS))\
#        .save_prefix(
#            lambda : f"adam_lr={float(LR)}_wd={float(WD)}_ls_{float(LS)}")\
#        .save_best()
#
#from pathlib import Path
#from mralign.viggo_eval import ViggoEval
#from plum2.tasks import TASKS
#for path in Path("experiments-viggo-v3/gru_inorder/train_gru_sgd/").glob("*"):
#    model_path = path / "model_checkpoints" / "optimal.pkl"
#    if (path / "model_checkpoints" / "optimal.pkl").exists():
#        t = ViggoEval(
#            model_path, va_batches_sm, utt_vcb)
#        TASKS['eval_val_' + path.name] = t
#
#
##for path in Path("experiments-viggo-v3/inorder/trainer_adam/").glob("*"):
##    model_path = path / "model_checkpoints" / "optimal.pkl"
##    if (path / "model_checkpoints" / "optimal.pkl").exists():
##        t = ViggoEval(
##            model_path, va_batches_sm, utt_vcb)
##        TASKS['eval_val_' + path.name] = t
#
#from mralign.viggopermtest import ViggoPermTest
#viggo_perm_test = ViggoPermTest([
#    "experiments-viggo-v2/train_inorder_baseline/model_checkpoints/optimal.pkl"], mr_vcb, utt_vcb, tr_ds.pipelines)
#from plum2.tasks import TASKS
#TASKS["perm_test_inorder"] = viggo_perm_test
#
#
#
