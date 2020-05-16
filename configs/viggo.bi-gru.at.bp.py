#!/usr/bin/env python


import argparse
import os
from pathlib import Path

from plum2 import seeds, work_dir
import plum2.tasks

from mrt.utils import (
    setup_vocab, setup_training_data, setup_validation_data,
    make_batches, setup_model, setup_trainer
)

import mrt.viggo.mr_utils as mr_utils


def main():
    parser = argparse.ArgumentParser(
        "Train Viggo transformer model (ls=aligned training, da=base)")
    parser.add_argument("output_dir", type=Path, help="save directory")
    parser.add_argument("--seed", type=int, default=234222452)
    parser.add_argument("--layers", default=2, type=int, help="num layers")
    parser.add_argument("--ls", default=0.1, type=float, 
                        help="label smoothing")
    parser.add_argument("--wd", default=0.0, type=float, help="weight decay")
    parser.add_argument("--opt", choices=["adam", "sgd"], default="adam",
                        help='optimizer')
    parser.add_argument("--lr", default=0.00001, type=float, 
                        help="learning rate")
    parser.add_argument("--tie-embeddings", action="store_true",
                        help="share decoder input/output embeddings")
    parser.add_argument("--attn", choices=['bahdanau', 'luong-general'],
                        default='bahdanau', help='attention type')
    parser.add_argument("--max-epochs", default=200, type=int,
                        help="max training epochs")
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--n-procs", default=2, type=int, 
                        help="num data loader processes")
    parser.add_argument("--tr-batch-size", default=128, type=int,
                        help="training batch size")
    parser.add_argument("--va-batch-size", default=1, type=int,
                        help="valid batch size")
    args = parser.parse_args()

    seeds(args.seed)
    assert os.getenv("MRT_EVAL_SCRIPT") is not None
    eval_script = os.getenv("MRT_EVAL_SCRIPT")

    dataset = "Viggo"
    lin_strat = "rule"
    is_delex = True

    mr_vcb, utt_vcb = setup_vocab(dataset, lin_strat, is_delex)
    tr_ds = setup_training_data(dataset, lin_strat, is_delex, mr_vcb, utt_vcb,
                                include_phrases=True)
    va_ds = setup_validation_data(dataset, lin_strat, is_delex,
                                  mr_vcb, utt_vcb)

    tr_batches = make_batches(tr_ds, args.tr_batch_size, args.n_procs, 
                              lin_strat, is_delex)
    va_batches = make_batches(va_ds, args.va_batch_size, args.n_procs, 
                              lin_strat, is_delex)
    
    model = setup_model("gru", "bi", args.layers, args.attn,
                        args.tie_embeddings, mr_vcb, utt_vcb, beam_size=8)


    trainer = setup_trainer(model, args.opt, args.lr, args.wd, args.ls, 
                            tr_batches, va_batches, args.max_epochs, 
                            utt_vcb, eval_script, mr_utils, 
                            lambda: f"Viggo/bi-gru/at/base+phrases/{args.seed}")

    env = {'proj_dir': args.output_dir, "gpu": args.gpu}
    trainer.run(env, verbose=True)


if __name__ == "__main__":
    main()
