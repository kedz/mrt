import argparse
from pathlib import Path
import json
import numpy as np
from collections import Counter
from subprocess import check_output, DEVNULL
from tempfile import NamedTemporaryFile
from pprint import pformat
import textwrap

import mrt.viggo.mr_utils
import mrt.viggo.rules
import mrt.e2e.mr_utils
import mrt.e2e.rules


def T2LMR(tags):
    tags = [t for i, t in enumerate(tags)
            if tags[i-1:i] != [t]]
    tags = [x for x in tags if x != '0']
    return tags

def prf(ref, predicted):
    
    ref_counts = Counter(ref)
    pred_counts = Counter(predicted)
    
    tp = 0
    for sf in set(list(ref) + list(predicted)):
        tp += min(ref_counts[sf], pred_counts[sf])

    recall = tp / max(1, len(ref))
    prec = tp / max(1, len(predicted))

    if (prec * recall) > 0:
        f1 = 2 * prec * recall / (prec + recall)
    else:
        f1 = 0.
    return prec, recall, f1

def bigrams(x):
    return list(zip(['@'] + x, x + ['@']))

def ord_prf(ref, predicted):
    return prf(bigrams(ref), bigrams(predicted))

def compute_autoeval(script_path, ref_path, pred_path):
    out = check_output([str(script_path), ref_path, pred_path],
                       stderr=DEVNULL)
    lines = out.decode('utf8').strip().split("\n")[-5:]

    results = {}
    for line in lines:
        m, v = line.split(": ")
        results[m] = float(v)

    return results


def rerank_beam(outputs, ref_linear_mr, mr_utils, rules):
    beam_scores = []
    for opt in outputs:
        pred_lmr = T2LMR(rules.tag_tokens(opt))
        beam_scores.append(prf(set(ref_linear_mr), pred_lmr)[2])
        
    return outputs[np.argmax(beam_scores)]

def evaluate(refs, in_fp, rule_set, eval_lin_strat, name, eval_log_dir):
    if rule_set == 'Viggo':
        mr_utils = mrt.viggo.mr_utils
        rules = mrt.viggo.rules
        delex = 'delex'
    elif rule_set == 'E2E':
        mr_utils = mrt.e2e.mr_utils
        rules = mrt.e2e.rules
        delex = 'lex'
    else:
        raise Exception(f"Bad rule set: {rule_set}")



    RULE_ORACLE = f'rule_{delex}'
    RULE_FREQ = f'freq_{delex}'
    RULE_MODEL = f'model_{delex}'

    
    all_results = {
        sch: {
            "semantics": [],
            "order": [],
        }
        for sch in ['greedy', 'beam', 'beam_rr']
    }

    if eval_log_dir != None:
        log_file = open(eval_log_dir / f'{name}.log', 'w')
    else:
        log_file = None

    pretty_outputs = {"greedy": [], 'beam': [], 'beam_rr': []}
    for i, (ref, pred_line) in enumerate(zip(refs, in_fp), 1):

        pred = json.loads(pred_line)

        canon_ref_lmr = mr_utils.remove_header(
            mr_utils.linearize_mr(ref['source']['mr'], delex=delex == 'delex'))

        eval_order = mr_utils.remove_header(
            ref['source']['sequence'][eval_lin_strat])

        greedy_toks = pred['outputs']['greedy']
        beam_toks = pred['outputs']['beam8'][0]
        beam_rr_toks = rerank_beam(pred['outputs']['beam8'], 
                                   canon_ref_lmr, mr_utils, rules)

        pretty_outputs['greedy'].append(
            mr_utils.lexicalize_string(
                mr_utils.detokenize(greedy_toks), 
                **ref['source']['mr']['slots']))
        pretty_outputs['beam'].append(
            mr_utils.lexicalize_string(
                mr_utils.detokenize(beam_toks), 
                **ref['source']['mr']['slots']))
        pretty_outputs['beam_rr'].append(
            mr_utils.lexicalize_string(
                mr_utils.detokenize(beam_rr_toks), 
                **ref['source']['mr']['slots']))

        greedy_lmr = T2LMR(rules.tag_tokens(greedy_toks))
        greedy_sem = prf(canon_ref_lmr, greedy_lmr)
        greedy_ord = ord_prf(eval_order, greedy_lmr)
        
        beam_lmr = T2LMR(rules.tag_tokens(beam_toks))
        beam_sem = prf(canon_ref_lmr, beam_lmr)
        beam_ord = ord_prf(eval_order, beam_lmr)

        beam_rr_lmr = T2LMR(rules.tag_tokens(beam_rr_toks))
        beam_rr_sem = prf(canon_ref_lmr, beam_rr_lmr)
        beam_rr_ord = ord_prf(eval_order, beam_rr_lmr)

        all_results['greedy']['semantics'].append(greedy_sem)
        all_results['greedy']['order'].append(greedy_ord)
        all_results['beam']['semantics'].append(beam_sem)
        all_results['beam']['order'].append(beam_ord)
        all_results['beam_rr']['semantics'].append(beam_rr_sem)
        all_results['beam_rr']['order'].append(beam_rr_ord)

        if log_file != None:
            print(f"Example {i}\n", file=log_file)
            print("MR: ", file=log_file)
            print(pformat(ref['source']['mr']), file=log_file)
            print("\nReferences:\n", file=log_file)

            ref_strings = ref['target']['reference_strings'].split('\n')
            for j, rs in enumerate(ref_strings, 1):
                print(textwrap.fill(f'({j}) {rs}', subsequent_indent='    '), 
                      file=log_file, end="\n\n")
            
            print("\nEncoder Input:\n", file=log_file)
            print(textwrap.fill(str(eval_order), 
                                initial_indent='    ',
                                subsequent_indent='    '),
                  end='\n\n', file=log_file)

            print((
                'Greedy:         (SEM P={:0.3f} R={:0.3f} F={:0.3f})'
                '  (ORD P={:0.3f} R={:0.3f} F={:0.3f})\n'
                ).format(*greedy_sem, *greedy_ord), file=log_file)
            print(textwrap.fill(pretty_outputs["greedy"][-1], 
                                initial_indent='    ', 
                                subsequent_indent='    '),
                  end="\n\n", file=log_file)

            print((
                'Beam:           (SEM P={:0.3f} R={:0.3f} F={:0.3f})'
                '  (ORD P={:0.3f} R={:0.3f} F={:0.3f})\n'
                ).format(*beam_sem, *beam_ord), file=log_file)

            print(textwrap.fill(pretty_outputs["beam"][-1], 
                                initial_indent='    ', 
                                subsequent_indent='    '),
                  end="\n\n", file=log_file)
            print((
                'Reranked Beam:  (SEM P={:0.3f} R={:0.3f} F={:0.3f})'
                '  (ORD P={:0.3f} R={:0.3f} F={:0.3f})\n'
                ).format(*beam_rr_sem, *beam_rr_ord), file=log_file)


            print(textwrap.fill(pretty_outputs["beam_rr"][-1], 
                                initial_indent='    ', 
                                subsequent_indent='    '),
                  end="\n\n", file=log_file)


            print("\n\n", file=log_file, flush=True)


#        for sch, toks in [('greedy', greedy_toks), ('beam', beam_toks), 
#                          ('beam_rr', beam_rr_toks)]:
#            pred_tags = rules.tag_tokens(toks)
#            pred_linear_mr = mr_utils.tags2linear_mr(pred_tags)
#            all_results[sch]['oracle_sem'].append(
#                prf(ref_oracle_mr, pred_linear_mr))
#            all_results[sch]['freq_sem'].append(
#                prf(ref_freq_mr, pred_linear_mr))
#            all_results[sch]['oracle_ord'].append(
#                ord_prf(ref_oracle_mr, pred_linear_mr))
#            all_results[sch]['freq_ord'].append(
#                ord_prf(ref_freq_mr, pred_linear_mr))
#
#            assert ref['source']['mr'] == pred['mr']
#
#   
    for sch in ['greedy', 'beam', 'beam_rr']:
        all_results[sch]['semantics'] = np.array(
            all_results[sch]['semantics']).mean(0)
        all_results[sch]['order'] = np.array(
            all_results[sch]['order']).mean(0)

    with NamedTemporaryFile("w") as ref_fp:
        for ref in refs:
            print(ref['target']['reference_strings'], end='\n\n', 
                  file=ref_fp, flush=True)
        for sch in pretty_outputs.keys():
            with NamedTemporaryFile("w") as pred_fp:
                print("\n".join(pretty_outputs[sch]), file=pred_fp, flush=True)
                r = compute_autoeval(
                    Path("../../v2/eval/e2e-metrics/measure_scores.py"),
                    ref_fp.name, pred_fp.name)
                all_results[sch]['auto'] = r
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rule_set', choices=['Viggo', 'E2E'])
    parser.add_argument('ref', type=Path)
    parser.add_argument("pred", type=str, nargs='+')
    parser.add_argument('--log-dir', type=Path, default=None)

    args = parser.parse_args()

    if args.log_dir != None:
        args.log_dir.mkdir(exist_ok=True, parents=True)

    with args.ref.open('r') as fp:
        refs = [json.loads(line) for line in fp]

    for arg in args.pred:
        path, eval_lin_strat, name = arg.split(',')
        
        with open(path, 'r') as fp:

            results = evaluate(refs, fp, args.rule_set, eval_lin_strat, name,
                               args.log_dir)

            print(name)
            for sch in ['greedy', 'beam', 'beam_rr']:
                print("{:35s}  {:0.3f}  {:0.3f}  {:0.3f}  {:0.3f}".format(
                    sch, 
                    results[sch]['auto']['BLEU'], 
                    results[sch]['auto']['ROUGE_L'], 
                    results[sch]['semantics'][2], 
                    results[sch]['order'][2]))



if __name__ == "__main__":
    main()
