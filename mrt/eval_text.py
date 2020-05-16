import argparse
from pathlib import Path
import json
import numpy as np
from collections import Counter, defaultdict
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

def sem_err(ref, predicted):
    ref = list(ref)
    added = []
    incorrect = []
    missing = []
    ref_slot2val = defaultdict(list)
    for sv in ref:
        s, v = sv.split("=")
        ref_slot2val[s].append(v)

    for sv in predicted:
        s,v = sv.split("=")
        if s not in ref_slot2val:
            added.append(sv)
        else:
            if v not in ref_slot2val[s]:
                incorrect.append(sv)
                #if s not in ['genres', 'platforms', 'player_perspectice']:
                #    if len(ref_slot2val[s]) > 0:
                #        ref_slot2val[s].pop(0)

            #else:
            #    ref_slot2val[s].pop(ref_slot2val[s].index(v))
    missing = [sf for sf in ref if sf not in predicted]
    #for s, vals in ref_slot2val.items():
    ##    for v in vals:
     #       missing.append(f'{s}={v}')

    total = len(added) + len(incorrect) + len(missing)
    counts = [len(missing), len(incorrect), len(added), total]

    return {"missing": missing, "incorrect": incorrect, 
            "added": added, "counts": counts}

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

def evaluate(script_path, refs, in_fp, rule_set, eval_lin_strat, name, 
             eval_log_dir):
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

        canon_ref_lmr = list(set(mr_utils.remove_header(
            ref['source']['sequence'][f'rule_{"delex" if delex == "delex" else "lex"}'])))

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

#        greedy_lmr = T2LMR(rules.tag_tokens(greedy_toks))
#        greedy_sem = prf(canon_ref_lmr, greedy_lmr)
#        greedy_ord = ord_prf(eval_order, greedy_lmr)
#        beam_lmr = T2LMR(rules.tag_tokens(beam_toks))
#        beam_sem = prf(canon_ref_lmr, beam_lmr)
#        beam_ord = ord_prf(eval_order, beam_lmr)

        beam_rr_lmr = T2LMR(rules.tag_tokens(beam_rr_toks))
        beam_rr_sem_err = sem_err(canon_ref_lmr, beam_rr_lmr)
        beam_rr_sem = prf(canon_ref_lmr, beam_rr_lmr)
        beam_rr_ord = ord_prf(eval_order, beam_rr_lmr)

        #all_results['greedy']['semantics'].append(greedy_sem)
        #all_results['greedy']['order'].append(greedy_ord)
        #all_results['beam']['semantics'].append(beam_sem)
        #all_results['beam']['order'].append(beam_ord)
        all_results['beam_rr']['semantics'].append(beam_rr_sem_err['counts'])
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

#?            print((
#?                'Greedy:         (SEM P={:0.3f} R={:0.3f} F={:0.3f})'
#?                '  (ORD P={:0.3f} R={:0.3f} F={:0.3f})\n'
#?                ).format(*greedy_sem, *greedy_ord), file=log_file)
#?            print(textwrap.fill(pretty_outputs["greedy"][-1], 
#?                                initial_indent='    ', 
#?                                subsequent_indent='    '),
#?                  end="\n\n", file=log_file)
#?
#?            print((
#?                'Beam:           (SEM P={:0.3f} R={:0.3f} F={:0.3f})'
#?                '  (ORD P={:0.3f} R={:0.3f} F={:0.3f})\n'
#?                ).format(*beam_sem, *beam_ord), file=log_file)
#?
#?            print(textwrap.fill(pretty_outputs["beam"][-1], 
#?                                initial_indent='    ', 
#?                                subsequent_indent='    '),
#?                  end="\n\n", file=log_file)
            print((
                'Reranked Beam:  (SEM P={:0.3f} R={:0.3f} F={:0.3f})'
                '  (ORD P={:0.3f} R={:0.3f} F={:0.3f})\n'
                ).format(*beam_rr_sem, *beam_rr_ord), file=log_file)
            print("              M: {}  I: {}  A: {}\n".format(*beam_rr_sem_err['counts'][:3]),
                file=log_file)
            print(textwrap.fill(pretty_outputs["beam_rr"][-1], 
                                initial_indent='    ', 
                                subsequent_indent='    '),
                  end="\n\n", file=log_file)

            print("Missing   : {}".format(', '.join(beam_rr_sem_err['missing'])), file=log_file)
            print("Incorrect : {}".format(', '.join(beam_rr_sem_err['incorrect'])), file=log_file)
            print("Added     : {}".format(', '.join(beam_rr_sem_err['added'])), file=log_file)

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
    for sch in ['beam_rr']:
        se = np.array(all_results[sch]['semantics'])
        print(se)
        se = 1.0 * np.array(all_results[sch]['semantics']).sum(0)
        se[3] = se[3] / len(all_results[sch]['semantics'])
        all_results[sch]['semantics'] = se
        all_results[sch]['order'] = np.array(
            all_results[sch]['order']).mean(0)

    with NamedTemporaryFile("w") as ref_fp:
        for ref in refs:
            print(ref['target']['reference_strings'], end='\n\n', 
                  file=ref_fp, flush=True)
        for sch in ['beam_rr']:
        #for sch in pretty_outputs.keys():
            with NamedTemporaryFile("w") as pred_fp:
                print("\n".join(pretty_outputs[sch]), file=pred_fp, flush=True)
                r = compute_autoeval(
                    str(script_path),
                    ref_fp.name, pred_fp.name)
                all_results[sch]['auto'] = r
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_script', type=Path)
    parser.add_argument('rule_set', choices=['Viggo', 'E2E'])
    parser.add_argument('ref', type=Path)
    parser.add_argument("pred", type=str, nargs='+')
    parser.add_argument("--autometrics", nargs="+", default=["BLEU", "ROUGE_L", "NIST", "METEOR", "CIDEr"])
    parser.add_argument('--log-dir', type=Path, default=None)

    args = parser.parse_args()

    if args.log_dir != None:
        args.log_dir.mkdir(exist_ok=True, parents=True)

    with args.ref.open('r') as fp:
        refs = [json.loads(line) for line in fp]

    all_results = []        
    for arg in args.pred:
        path, eval_lin_strat, name, model_type, lin_strat, = arg.split(',')

        with open(path, 'r') as fp:

            results = evaluate(args.eval_script, refs, fp, args.rule_set, 
                               eval_lin_strat, name, args.log_dir)
            all_results.append({"model_type": model_type, "lin_strat": lin_strat, "results": results})
            print("{:20}  {:10s}  {:4s}  {:4s}  {:4s}  {:4s}  {:4s}  {:4s}  {:4s}".format(
                "model", "search", "BLEU", "METEOR", "NIST", "CIDEr", "ROUGE-L", "SEM", "ORD"))
            for sch in ['beam_rr']: #['greedy', 'beam', 'beam_rr']:
                print("{:20s}  {:10s}  {:0.3f}  {:0.3f}  {:0.3f}  {:0.3f}  {:0.3f}  {:0.3f}  {:0.3f}".format(
                    name,
                    sch, 
                    results[sch]['auto']['BLEU'], 
                    results[sch]['auto']['METEOR'], 
                    results[sch]['auto']['NIST'], 
                    results[sch]['auto']['CIDEr'], 
                    results[sch]['auto']['ROUGE_L'], 
                    results[sch]['semantics'][3], 
                    results[sch]['order'][2]))

    model_type_order = ['BiGru', 'Transformer']
    lin_strat_order = ['Rand', 'IncFreq', 'IncFreqFixed', 'Align (F)', 'Align (M)', 'Align (O)']

    all_results.sort(key=lambda x: lin_strat_order.index(x['lin_strat']))
    all_results.sort(key=lambda x: model_type_order.index(x['model_type']))

    
    header = "    Model & Lin. Str. & " + " & ".join(args.autometrics) 
    header += " & M & I & A & Err"
    header += " & Ord."
    header += r" \\"
    header = header.replace("ROUGE_L", "RG-L")
    header = header.replace("METEOR", "MET.")

    cols = len(args.autometrics) + 4 + 1
    
    model_type_counts = Counter([r['model_type'] for r in all_results])
    print(r"\begin{tabular}{ll" + "c" * cols + "}")
    print(r"    \toprule")
    print(r" & & \multicolumn{" +str(len(args.autometrics)) +r"}{c}{Automatic Metrics} & \multicolumn{4}{c}{Semantic} & \\")
    print(header)
    has_mtype = set()
    for result in all_results:
        if result['model_type'] in has_mtype:
            mtype = ''
        else:
            print(r"\midrule")
            mtype = (
                r'\multirow{' 
                + str(model_type_counts[result['model_type']]) 
                + r"}{*}{\textsc{" + result['model_type'] + "}}"
            )
            has_mtype.add(result['model_type'])
        print(mtype, "&", result['lin_strat'], "&", 
            " & ".join([f'{result["results"]["beam_rr"]["auto"][m]:0.3f}' for m in args.autometrics]),
            " & " + " & ".join([f'{int(x)}' for x in result['results']['beam_rr']['semantics'][:3]]),
            f" & {result['results']['beam_rr']['semantics'][3] * 100:0.3f}",
            " & {:0.3f}".format(result['results']['beam_rr']['order'][2]) if result['lin_strat'].startswith("Alig") else "& --",
            r" \\")

    print(r"    \bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    main()
