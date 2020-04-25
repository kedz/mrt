import torch

import plum2
from plum2.loss import ClassCrossEntropy
from plum2.metrics import Script
from plum2.optimizer import SGD, Adam
from plum2.logger import ModelOutputLogger

from mrt.generate_text import GenerateText
import mrt.viggo.mr_utils
import mrt.e2e.mr_utils


def setup_vocab(dataset, lin_strat, delex, datadir='data'):
    src_seq = f"{lin_strat}_{'delex' if delex else 'lex'}" 
    tgt_seq = f"{'delex' if delex else 'lex'}"

    part = 'train.jsonl' if dataset == 'Viggo' else 'train.no-ol.jsonl' 

    with plum2.dataset(f"{dataset}.vocab") as ds:
        ds.jsonl(f"{datadir}/{dataset}/{dataset}.{part}")

        with ds.vocab("mr") as mr_vcb:
            mr_vcb.apply_indexers(['source', 'sequence', src_seq])\
                .start().stop()

            if 'fixed' not in lin_strat:
                mr_vcb.pad()

            phrases = ["NP", "VP", "ADJP", "ADVP", "PP", "S", "SBAR"]
            mr_vcb.add_tokens([
                tag
                for nt in phrases
                for tag in [f'<sos-{nt}>', f'<eos-{nt}>']
            ])

        with ds.vocab("utt") as utt_vcb:
            utt_vcb.apply_indexers(['target', "sequence", tgt_seq]).pad()\
                .start().stop().unk().at_least(3)

    return mr_vcb, utt_vcb

def setup_training_data(dataset, lin_strat, delex, mr_vcb, utt_vcb, 
                        datadir='data', include_phrases=False,
                        include_templates=False):

    if dataset == 'Viggo':
        mr_utils = mrt.viggo.mr_utils
    elif dataset == 'E2E':
        mr_utils = mrt.e2e.mr_utils
    else:
        raise Exception(f"Bad dataset: {dataset}")

    part = 'train.jsonl' if dataset == 'Viggo' else 'train.no-ol.jsonl' 


    src_seq = f"{lin_strat}_{'delex' if delex else 'lex'}" 
    tgt_seq = f"{'delex' if delex else 'lex'}"

    with plum2.dataset(f"{dataset}.train") as ds:
        orig_ds = ds.jsonl(f"{datadir}/{dataset}/{dataset}.{part}")

        extra_data = []
        if include_phrases:
            extra_data.append(
                ds.jsonl(f"{datadir}/{dataset}/{dataset}.phrases.no-ol.jsonl"))
        if include_templates:
            extra_data.append(
                ds.jsonl(
                    f"{datadir}/{dataset}/{dataset}.templates.no-ol.jsonl"))
        if len(extra_data) > 0:
            ds.concat(orig_ds, *extra_data)

        def encoder_input_getter(ex):

            if lin_strat == 'random':
                tokens = mr_utils.linearize_mr(ex['source']['mr'],
                                               order='random', delex=delex)
            else:
                tokens = ex['source']['sequence'][src_seq]

            if 'phrase' in ex['source']:
                phrase = ex['source']['phrase']
                tokens = [f'<sos-{phrase}>'] + tokens + [f'<eos-{phrase}>']
            else:
                tokens = [f'<sos>'] + tokens + [f'<eos>']
            return tokens

        with ds.feature("encoder_input") as x:
            x.apply_func(encoder_input_getter).lookup(mr_vcb)\
                .batch_varlen_seq(mr_vcb.pad_index)

        with ds.feature("decoder_input") as y:
            y.apply_indexers(['target', "sequence", tgt_seq]).lookup(utt_vcb)\
                .prepend(utt_vcb.start_index)\
                .batch_varlen_seq(utt_vcb.pad_index)

        with ds.feature("target") as y:
            y.apply_indexers(["target", "sequence", tgt_seq]).lookup(utt_vcb)\
                .extend(utt_vcb.stop_index)\
                .batch_varlen_seq(utt_vcb.pad_index)
    return ds 

def setup_validation_data(dataset, lin_strat, delex, mr_vcb, utt_vcb, 
                          datadir='data'):

    src_seq = f"{lin_strat}_{'delex' if delex else 'lex'}" 
    tgt_seq = f"{'delex' if delex else 'lex'}"
    part = 'valid.jsonl' if dataset == 'Viggo' else 'valid.no-ol.jsonl' 

    with plum2.dataset(f"{dataset}.valid") as ds:
        ds.jsonl(
            f"{datadir}/{dataset}/{dataset}.{part}"
        )
        
        # Include original MR dictionary for use in re-lexicalizing decoder
        # output. Only really necessary when trainin on delexicalized data.
        with ds.feature("mr") as mr:
            mr.apply_indexers(['source', 'mr'])

        # Include pretty string versions of input and output for debugging 
        # and logging. 
        with ds.feature("encoder_input_pretty") as x_pretty:
            x_pretty.apply_indexers(['source', 'sequence', src_seq])\
                .apply_func(lambda x: ','.join(x))

        with ds.feature("target_pretty") as y_pretty:
            y_pretty.apply_indexers(["target", "reference"])

        # The encoder input sequence batched (and padded when necessary).
        with ds.feature("encoder_input") as x:
            x.apply_indexers(['source', 'sequence', src_seq]).lookup(mr_vcb)\
                .prepend(mr_vcb.start_index).extend(mr_vcb.stop_index)\
                .batch_varlen_seq(mr_vcb.pad_index)

        # Decoder input and target output sequences batched and padded.
        with ds.feature("decoder_input") as y_in:
            y_in.apply_indexers(['target', "sequence", tgt_seq])\
                .lookup(utt_vcb)\
                .prepend(utt_vcb.start_index)\
                .batch_varlen_seq(utt_vcb.pad_index)

        with ds.feature("target") as y_out:
            y_out.apply_indexers(["target", "sequence", tgt_seq])\
                .lookup(utt_vcb)\
                .extend(utt_vcb.stop_index)\
                .batch_varlen_seq(utt_vcb.pad_index)

    return ds 

def make_batches(ds, batch_size, workers, lin_strat, is_delex):
    if "fixed" in lin_strat:
        sort = False
        sort_key = None
    elif lin_strat == 'random':
        sort = True
        src_seq = f"inc_freq_{'delex' if is_delex else 'lex'}" 
        sort_key = lambda x: len(x['source']['sequence'][src_seq])
    else:
        src_seq = f"{lin_strat}_{'delex' if is_delex else 'lex'}" 
        sort = True
        sort_key = lambda x: len(x['source']['sequence'][src_seq])

    return ds.batch_iter(batch_size, sort=sort, sort_key=sort_key, 
                         workers=workers)

def setup_model(arch, rnn_dir, layers, rnn_attn, weight_tying,
                mr_vocab, utt_vocab, beam_size=4):
    if arch in ['gru', 'lstm']:
        if rnn_dir == 'bi':
            with plum2.seq2seq('rnn') as m:
                m.arch(arch).emb(512).hidden(512).layers(layers)\
                    .bi().dropout(0.1).attn(rnn_attn)\
                    .enc_vocab(mr_vocab).dec_vocab(utt_vocab)\
                    .beam_search(max_steps=100, beam_size=beam_size)\
                    .tie_decoder_embeddings(weight_tying)
        else:
            with plum2.seq2seq('rnn') as m:
                m.arch(arch).emb(512).hidden(512).layers(layers)\
                    .pass_thru_bridge().dropout(0.1).attn(rnn_attn)\
                    .enc_vocab(mr_vocab).dec_vocab(utt_vocab)\
                    .beam_search(max_steps=100, beam_size=beam_size)\
                    .tie_decoder_embeddings(weight_tying)
    else:
        with plum2.seq2seq('transformer') as m:
            m.arch("transformer").emb(512).hidden(2048).layers(layers)\
                .dropout(0.1)\
                .enc_vocab(mr_vocab).dec_vocab(utt_vocab)\
                .beam_search(max_steps=100, beam_size=beam_size)\
                .tie_decoder_embeddings(weight_tying)

    return m

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
    def setup_optimizer(self, trainer, verbose=False):
        self.optimizer = torch.optim.Adam(
            trainer.model.parameters(),
            lr = 0, betas=(0.9, 0.98), eps=1e-9)

    def zero_grad(self):
        self.optimizer.zero_grad()

def setup_hps_trainer(model, optname, learning_rate, weight_decay, 
                      label_smoothing, 
                      train_batches, valid_batches, max_steps, dec_vocab,
                      eval_script, mr_utils, save_prefix):

    if optname == 'sgd':
        opt = SGD(learning_rate, weight_decay=weight_decay)
    elif optname == 'adam':
        opt = Adam(learning_rate, weight_decay=weight_decay)
    elif optname == 'adamtri':
        opt = NoamOpt(512, 1.0, 8000,None)
    else:
        raise Exception(f"Bad optimizer {opt}") 

    loss = ClassCrossEntropy(
        padding_index=dec_vocab.pad_index,
        label_smoothing=label_smoothing)

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

    # Setup eval script.
    def get_reference(m, b, fs): 
        return [f"{x}\n" for x in b["target_pretty"]]
    
    def get_predicted(m, b, fs): 
        outputs = [] 
        for mr, tokens in zip(b['mr'], fs["search"]["beam"].output()):
            detok = mr_utils.detokenize(tokens)
            output = mr_utils.lexicalize_string(detok, **mr['slots'])
            outputs.append(output)
        return outputs

    def parse_result_string(result_text):
        results = {}
        lines = result_text.decode('utf8').strip().split("\n")[-5:]
        for line in lines:
            metric, value = line.split(": ")
            results[metric] = float(value)
        return results
        
    EVERY = 25
    metrics = Script(
        eval_script,
        get_reference,
        get_predicted,
        parse_result_string,
        lambda x: x["BLEU"], 
        False,
        every=EVERY
    )
    logger = ModelOutputLogger(
        lambda m, b, fs: b["encoder_input_pretty"],
        get_predicted,
        target_getter=get_reference,
        every_epoch=EVERY,
    )

    orig_batch_size = valid_batches.batch_size
    def reduce_batch_size_on_search(epoch, trainer, results):
        if (epoch + 1) % EVERY == 0:
            trainer.valid_dataset.batch_size = 1
        else:
            trainer.valid_dataset.batch_size = orig_batch_size

    with plum2.trainer("hps_train") as trainer:
        trainer.model(model).optim(opt)\
            .train(train_batches).valid(valid_batches)\
            .epochs(max_steps)\
            .loss(loss)\
            .add_callback(reduce_batch_size_on_search)\
            .valid_metrics(metrics)\
            .valid_logger(logger)\
            .save_prefix(save_prefix)
            
        if optname != 'adamtri':
            trainer.add_callback(update_lr)

    return trainer

def setup_trainer(model, optname, learning_rate, weight_decay, label_smoothing,
                  train_batches, valid_batches, max_steps, dec_vocab,
                  eval_script, mr_utils, save_prefix, train_suffix):

    if optname == 'sgd':
        opt = SGD(learning_rate, weight_decay=weight_decay)
    elif optname == 'adam':
        opt = Adam(learning_rate, weight_decay=weight_decay)
    elif optname == 'adamtri':
        opt = NoamOpt(512, 1.0, 8000,None)
    else:
        raise Exception(f"Bad optimizer {opt}") 

    loss = ClassCrossEntropy(
        padding_index=dec_vocab.pad_index,
        label_smoothing=label_smoothing)

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

    # Setup eval script.
    def get_reference(m, b, fs): 
        return [f"{x}\n" for x in b["target_pretty"]]
    
    def get_predicted(m, b, fs): 
        outputs = [] 
        for mr, tokens in zip(b['mr'], fs["search"]["beam"].output()):
            detok = mr_utils.detokenize(tokens)
            output = mr_utils.lexicalize_string(detok, **mr['slots'])
            outputs.append(output)
        return outputs

    def parse_result_string(result_text):
        results = {}
        lines = result_text.decode('utf8').strip().split("\n")[-5:]
        for line in lines:
            metric, value = line.split(": ")
            results[metric] = float(value)
        return results
        
    EVERY = 1
    metrics = Script(
        eval_script,
        get_reference,
        get_predicted,
        parse_result_string,
        lambda x: x["BLEU"], 
        False,
        every=EVERY
    )
    logger = ModelOutputLogger(
        lambda m, b, fs: b["encoder_input_pretty"],
        get_predicted,
        target_getter=get_reference,
        every_epoch=EVERY,
    )

    with plum2.trainer("train_" + train_suffix) as trainer:
        trainer.model(model).optim(opt)\
            .train(train_batches).valid(valid_batches)\
            .epochs(max_steps)\
            .loss(loss)\
            .valid_metrics(metrics)\
            .valid_logger(logger)\
            .save_prefix(save_prefix)\
            .save_best()
            
        if optname != 'adamtri':
            trainer.add_callback(update_lr)

    return trainer

def setup_generator(model_path, dataset, encoder_vocab, decoder_vocab, order,
                    output_path, task_name):
    from plum2.tasks import TASKS
    TASKS[task_name] = GenerateText(model_path, dataset, encoder_vocab, 
                                    decoder_vocab, order, output_path)

