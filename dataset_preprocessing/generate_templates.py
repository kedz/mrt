#!/usr/bin/env python

import argparse
from pathlib import Path
import json
from collections import defaultdict
from itertools import permutations, product

import mrt.e2e.mr_utils
import mrt.e2e.template_meta
import mrt.viggo.mr_utils
import mrt.viggo.template_meta


def get_rule_set(rule_set):
    if rule_set == "E2E":
        mr_utils = mrt.e2e.mr_utils
        template_meta = mrt.e2e.template_meta
    elif rule_set == 'Viggo':
        mr_utils = mrt.viggo.mr_utils
        template_meta = mrt.viggo.template_meta
    else:
        raise Exception(f'Bad rule set: {rule_set}')
    return mr_utils, template_meta
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rule_set", choices=['Viggo', 'E2E'])
    parser.add_argument("freqs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    freq_info = json.loads(args.freqs.read_text())
    mr_utils, template_meta = get_rule_set(args.rule_set) 
    
    ALL_SLOTS = list(template_meta.slot_lexical_items.keys())

    total = 0
    with args.output.open("w") as fp:
        for s1, s2 in permutations(ALL_SLOTS, 2):
            if s1 == s2:
                continue

            type1 = s1.split("=")[0]
            type2 = s2.split("=")[0]

            if (type1, type2) in template_meta.invalid_transitions:
                continue

            print(" " * 78 + "\r" + str((s1, s2)), end='\r', flush=True)
            for template in template_meta.templates[(type1, type2)]:
                (s1_form, s2_form), template_string = template
                s1_opts = template_meta.slot_lexical_items[s1].get(s1_form, [])
                s2_opts = template_meta.slot_lexical_items[s2].get(s2_form, [])
                for s1_lex, s2_lex in product(s1_opts, s2_opts):
                    result = str(template_string)
                    result = result.replace(s1_form + "_1", s1_lex)
                    result = result.replace(s2_form + "_2", s2_lex)
                    mr_seq = [s1, s2]
                    mr = mr_utils.linear_mr2mr(mr_seq)
                    mr['da'] = "inform" 
                    tokens = mr_utils.tokenize(result)
                    assert tokens[-1] == "."
                    for t in tokens:
                        if t.isupper():
                            assert mr_utils.is_placeholder(t)


                    if args.rule_set == 'Viggo':
                        seq_key = 'delex'
                    else:
                        seq_key = 'lex'

                    example = {
                        "source": {
                            "mr": mr,
                            "sequence": {
                                f'rule_{seq_key}': \
                                    mr_utils.mr2header(mr) + mr_seq,
                                f'inc_freq_{seq_key}': \
                                    mr_utils.linearize_mr(
                                        mr, delex=seq_key=='delex',
                                        order='inc_freq', freq_info=freq_info),
                                f'dec_freq_{seq_key}': \
                                    mr_utils.linearize_mr(
                                        mr, delex=seq_key=='delex',
                                        order='dec_freq', freq_info=freq_info),
                                f'inc_freq_fixed_{seq_key}': \
                                    mr_utils.linearize_mr(
                                        mr, delex=seq_key=='delex',
                                        order='inc_freq_fixed', 
                                        freq_info=freq_info),
                                f'dec_freq_fixed_{seq_key}': \
                                    mr_utils.linearize_mr(
                                        mr, delex=seq_key=='delex',
                                        order='dec_freq_fixed',
                                        freq_info=freq_info),
                            }
                        },
                        "target": {
                            "sequence": {seq_key: tokens,}, 
                            'reference': ' '.join(tokens)
                        },
                    }
                    print(json.dumps(example), file=fp)
                    total += 1
    print(" " * 78, end='\r')
    print(f'Generated {total} transitions.')


if __name__ == '__main__':
    main()
