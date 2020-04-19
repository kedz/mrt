#!/usr/bin/env python

import argparse
from pathlib import Path
import csv
import json
import re

import mrt.e2e.mr_utils
import mrt.e2e.rules 
import mrt.viggo.mr_utils
import mrt.viggo.rules 


def check_sfseq(gt, pred):
    gt = list(gt)
    pred = list(pred)
    if len(gt) != len(pred):
        return False

    for s in pred:
        if s == 'customer_rating=low':
            if not (s in gt or 'customer_rating=1 out of 5' in gt):
                return False
        elif s == 'customer_rating=1 out of 5':
            if not (s in gt or 'customer_rating=low' in gt):
                return False
        elif s == 'customer_rating=average':
            if not (s in gt or 'customer_rating=3 out of 5' in gt):
                return False
        elif s == 'customer_rating=3 out of 5':
            if not (s in gt or 'customer_rating=average' in gt):
                return False
        elif s == 'customer_rating=high':
            if not (s in gt or 'customer_rating=5 out of 5' in gt):
                return False
        elif s == 'customer_rating=5 out of 5':
            if not (s in gt or 'customer_rating=high' in gt):
                return False


        elif s == 'price_range=less than £20':
            if not (s in gt or 'price_range=cheap' in gt):
                return False
        elif s == 'price_range=cheap':
            if not (s in gt or 'price_range=less than £20' in gt):
                return False

        elif s == 'price_range=more than £30':
            if not (s in gt or 'price_range=high' in gt):
                return False
        elif s == 'price_range=high':
            if not (s in gt or 'price_range=more than £30' in gt):
                return False


        elif s == 'price_range=£20-25':
            if not (s in gt or 'price_range=moderate' in gt):
                return False
        elif s == 'price_range=moderate':
            if not (s in gt or 'price_range=£20-25' in gt):
                return False
        
        elif s not in gt:
            return False

    return True

def format_example(mr_line, utt_line, rule_set, exid, history):

    if rule_set == 'E2E':
        mr_utils = mrt.e2e.mr_utils
        rules = mrt.e2e.rules
    elif rule_set == 'Viggo':
        mr_utils = mrt.viggo.mr_utils
        rules = mrt.viggo.rules
    else:
        raise Exception(f"Bad rule set: {rule_set}")

    # Get the reference string (for BLEU/ROUGE/etc. evals).
    ref = mr_utils.correct_utterance(utt_line)
    # Get dict MR representation. 
    mr = mr_utils.extract_mr(mr_line)

    # Tokenize reference for target sequence with and without delexicalization.
    tokens_lex = mr_utils.tokenize(ref.lower())
    tokens_delex = mr_utils.delexicalize_tokens(tokens_lex, **mr['slots'])

    # Detokenize target sequence to verify the correctness of the detokenizer
    # (with/without delexicalization).
    detok_ref = mr_utils.detokenize(tokens_lex)
    detok_ref_delex = mr_utils.detokenize(tokens_delex)

    # Tag lexicalized and delexicalized token sequences. 
    tags_lex = rules.tag_tokens(tokens_lex, **mr['slots'])
    tags_delex = rules.tag_tokens(tokens_delex, **mr['slots'])
    print(tokens_lex)
    for tag, tok in zip(tags_lex, tokens_lex):
        print(f"{tag:40s}  {tok}")
    print()

    # Convert lexicalized tag sequence to linear mr.
    rule_linear_mr = sorted(set(mr_utils.tags2linear_mr(tags_lex)))

    # If the ground truth MR is incorrect, use the corrected version.
    # Otherwise use dataset provided MR.
    # In either case convert it to linearized MR. 
    if exid in history:
        gt_linear_mr = sorted(set(history[exid]))
    else:    
        gt_linear_mr = mr_utils.linearize_mr(mr, delex=False, 
                                             return_header=False)
    
    # Compare linear mr of tagger to ground truth.
    linear_mr_pair = {
        "ground_truth": gt_linear_mr, 
        "predicted": rule_linear_mr
    }

    # Compare delexicalized and lexicalized linear mrs. They should be the 
    # same (after replacing the lexicalzed fillers with PLACEHOLDER).
    linear_mr_delex_pair = {
        "lex": mr_utils.tags2linear_mr(tags_lex, delex=True), 
        "delex": mr_utils.tags2linear_mr(tags_delex),
    }

    # Compare detokenized strings with and without delexicalization.
    # They should be the same after relexicalizing the detokenized delexical
    # token sequence.
    pr_mr = mr_utils.tags2mr(tags_lex)
    pr_mr['slots']['specifier'] = mr['slots'].get('specifier', None)
    detok_pair = {
        "lex": detok_ref,
        "delex": mr_utils.lexicalize_string(detok_ref_delex, **pr_mr['slots']),
    }

    # Compare detokenized string to original reference string. Should be equal.
    ref_string_pair = {
        "reference": ref,
        "detokenized": detok_ref,
    }

    return (linear_mr_pair, linear_mr_delex_pair, detok_pair, ref_string_pair)

def compare(pred_mr, gt_slots):
    gt_slots = list(gt_slots)
    sf_seq = list(pred_mr)
    return check_sfseq(set(gt_slots), set(sf_seq))


def verify_partition(input_path, save_path, rule_set, history):

    with input_path.open('r') as in_fp:
        reader = csv.reader(in_fp)
        next(reader)
        for exid, item in enumerate(reader):
            mr_line, utt_line = item[:2]
#            if exid < 15088:
#                continue
            print(exid)
            print(mr_line)
            print(utt_line)
                
            X = format_example(mr_line, utt_line, rule_set, exid, history)
            linear_mr_pair, linear_mr_delex_pair, detok_pair, ref_str_pair = X
            
            recheck = False
            while not compare(linear_mr_pair["predicted"], 
                              linear_mr_pair['ground_truth']) \
                    or (
                        set(linear_mr_delex_pair['lex'])
                            != set(linear_mr_delex_pair['delex'])) \
                    or recheck:

#            while not compare(linear_mr_pair["predicted"], 
#                              linear_mr_pair['ground_truth']) \
#                    or (
#                        linear_mr_delex_pair['lex'] 
#                            != linear_mr_delex_pair['delex']) \
#                    or detok_pair['lex'] != detok_pair['delex'] \
#                    or (
#                        ref_str_pair['reference'] 
#                            != ref_str_pair['detokenized']) \
#                    or recheck:



                print()
                print("Reference String")
                print(ref_str_pair['reference'])
                print("Detokenized String")
                print(ref_str_pair['detokenized'])
                print()
                print('Rule MR')
                print(sorted(set(linear_mr_pair['predicted'])))
                print("Original MR")
                print(sorted(set(linear_mr_pair['ground_truth'])))
                print()
                print("Rule MR -> Delex")
                print(sorted(set(linear_mr_delex_pair['lex'])))
                print("Delex -> Rule MR")
                print(sorted(set(linear_mr_delex_pair['delex'])))
#                print()
#                print("Detokenized (Lexical)")
#                print(detok_pair['lex'])
#                print("Detokenized Lexicalized (Delexical)")
#                print(detok_pair['delex'])
#
                response = input()
                if response == 'a':
                    save_item = sorted(set(linear_mr_pair["predicted"]))
                    history[exid] = save_item
                    with save_path.open('a+') as fp:
                        print(json.dumps([exid, save_item]),
                            flush=True, file=fp)
                    X = format_example(mr_line, utt_line, rule_set, 
                                       exid, history)
                    (linear_mr_pair, linear_mr_delex_pair, detok_pair, 
                     ref_str_pair) = X

                if response == 'e':
                    try:
                        import importlib
                        importlib.reload(mrt.e2e.mr_utils)
                        importlib.reload(mrt.e2e.rules)
                        importlib.reload(mrt.viggo.mr_utils)
                        importlib.reload(mrt.viggo.rules)
                    except Exception as e:
                        print(e)
            
                    X = format_example(mr_line, utt_line, rule_set, 
                                       exid, history)
                    (linear_mr_pair, linear_mr_delex_pair, detok_pair, 
                     ref_str_pair) = X
                    recheck = True 

                if response == 'n':
                    recheck = False
                    X = format_example(mr_line, utt_line, rule_set, 
                                       exid, history)
                    (linear_mr_pair, linear_mr_delex_pair, detok_pair, 
                     ref_str_pair) = X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rule_set', choices=['Viggo', 'E2E'])
    parser.add_argument("load", type=Path)
    parser.add_argument("save", type=Path)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()

    history = {}
    if args.load.exists():
        with args.load.open('r') as fp:
            for line in fp:
                exid, mr = eval(line)
                history[exid] = mr
    if args.load != args.save:
        with args.save.open('w') as save_fp:
            for key in sorted(history.keys()):
                print(json.dumps([key, history[key]]), file=save_fp, flush=True)
    verify_partition(args.input, args.save, args.rule_set, history) 

if __name__ == '__main__':
    main()
