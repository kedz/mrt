
import argparse
import json
from pathlib import Path
from collections import defaultdict
import math
import numpy as np
from itertools import permutations
from collections import Counter
from mrt.viggo.tagging_rules import get_specifier_feats
import shutil
import tempfile
import random

random.seed(917601650175)


def example_iter(path):
    with path.open("r") as fp:
        for line in fp:
            yield json.loads(line)

def bigram_iter(items):
    return zip(['@'] + list(items), list(items) + ['@'])

def lm_score(items, lm):
    return sum(lm[t1][t2] for t1, t2 in bigram_iter(items))
    
def get_freq_mappings(path, seq_key):
    unifreq = defaultdict(int)
    freq = defaultdict(lambda : defaultdict(int))
    lm = defaultdict(lambda : defaultdict(int))

    vocab = set(['@'])

    for ex in example_iter(path):
        oracle_order = tuple(ex['source']['sequence'][seq_key][2:])
        canon_rep = tuple(sorted(oracle_order))
        freq[canon_rep][oracle_order] += 1

        for sf in set(oracle_order):
            unifreq[sf] += 1

        for t1,t2 in bigram_iter(oracle_order):
            lm[t1][t2] += 1
            vocab.add(t1)
            vocab.add(t2)
    
    for t1 in vocab:
        norm = 0
        for t2 in vocab:
            norm += lm[t1][t2] + 1e-6
        log_norm = math.log(norm)
        for t2 in vocab:
            lm[t1][t2] = math.log(lm[t1][t2] + 1e-6) - log_norm

    co2sfo = {}
    for canon_order, freqs in freq.items():
        max_freq = max(freqs.values())
        
        max_items = [item for item in freqs.items() if item[1] == max_freq]
        if len(max_items) == 1:
            co2sfo[canon_order] = max_items[0][0]
        else:
            lm_scores = [lm_score(item[0], lm) for item in max_items]
            I = np.argsort(lm_scores)            
            co2sfo[canon_order] = max_items[I[-1]][0]

    return co2sfo, lm, unifreq

def add_orderings(path, sf_map, sf_lm, sf_freq, s_map, s_lm, 
                  make_inc_order_fixed, make_dec_order_fixed, orig=False):
    
    suffix = '' if not orig else '_orig'
    sf_count = 0
    s_count = 0
    g_count = 0
    with tempfile.NamedTemporaryFile('w') as tmp_file:
        for ex in example_iter(path):


            oracle_sf_order = tuple(ex['source']['sequence']['oracle' + suffix][2:])
            canon_sf_rep = tuple(sorted(oracle_sf_order))

            oracle_s_order = tuple(ex['source']['sequence']['oracle_slots' + suffix][2:])
            canon_s_rep = tuple(sorted(oracle_s_order))

            if canon_sf_rep in sf_map:
                freq_order = sf_map[canon_sf_rep]
                sf_count += 1
            elif canon_s_rep in s_map:
                s_order = s_map[canon_s_rep]
                freq_order = find_best_constrained_perm(
                    oracle_sf_order, s_order, sf_lm)
                s_count += 1
            else:
                freq_order = generate_greedy_lm(oracle_sf_order, sf_lm)
                g_count += 1
            ex['source']['sequence']['frequency_model'+ suffix] = (
                ex['source']['sequence']['oracle'+suffix][:2] + list(freq_order)
            )

            random_order = list(set(oracle_sf_order))
            random.shuffle(random_order)
            ex['source']['sequence']['random'+ suffix] = (
                ex['source']['sequence']['oracle'+suffix][:2] + random_order
            )

            inc_order = sorted(random_order, key=lambda x: sf_freq[x])
            dec_order = sorted(random_order, key=lambda x: sf_freq[x], reverse=True)

            ex['source']['sequence']['inc_order'+ suffix] = (
                ex['source']['sequence']['oracle'+suffix][:2] + inc_order
            )

            ex['source']['sequence']['dec_order'+ suffix] = (
                ex['source']['sequence']['oracle'+suffix][:2] + dec_order
            )

            inc_order_fixed = make_inc_order_fixed(ex['source']['mr' + suffix])
            ex['source']['sequence']['inc_order_fixed'+ suffix] = \
                inc_order_fixed
            dec_order_fixed = make_dec_order_fixed(ex['source']['mr' + suffix])
            ex['source']['sequence']['dec_order_fixed'+ suffix] = \
                dec_order_fixed

            print(json.dumps(ex), file=tmp_file)

        tmp_file.flush()
        shutil.copy2(tmp_file.name, str(path)) 
        
    print(sf_count, s_count, g_count)

COUNT = 0
LIST_SLOTS = ['platforms', 'genres', 'player_perspective']
def add_uncorrected_orderings(path, sf_map, sf_lm, s_map, s_lm):
    global COUNT

    with tempfile.NamedTemporaryFile('w') as tmp_file:
        for ex in example_iter(path):

            correct_order = tuple(ex['source']['sequence']['oracle'][2:])

            correct_order_keys = [
                x.split('=')[0] if x.split('=')[0] not in LIST_SLOTS else x
                for x in correct_order]
            
            orig_oracle = []
            for k, v in ex['source']['mr_orig']['slots'].items():
                if k == 'rating':
                    continue
                elif isinstance(v, list):
                    for vi in v:
                        orig_oracle.append(f'{k}={vi}')
                elif k in ['name', 'developer', 'release_year', 'exp_release_data']:
                    orig_oracle.append(f'{k}=PLACEHOLDER')
                else:
                    orig_oracle.append(f'{k}={v}')
                
            orig_oracle_keys = [x.split('=')[0] if x.split('=')[0] not in ['platforms', 'genres', 'player_perspective'] else x
                                for x in orig_oracle]

            assert all([x in correct_order_keys for x in orig_oracle_keys ])
            

            # sometimes there are duplicate mrs, e.g. "NAME is an rpg. NAME is cool." so we need to number 
            # slots to get consistent ordering.
            correct_order_counts = Counter(correct_order_keys)
            correct_order_keys_ids = [x + "#" + str(sum([y == x for y in correct_order_keys[:i]])) for i, x in enumerate(correct_order_keys)]

            correct_order_idxs = {x: i for i, x in enumerate(correct_order_keys_ids)}

            for key in list(orig_oracle_keys):
                lim = correct_order_counts[key]
                for i in range(1, correct_order_counts[key]):
                    orig_oracle_keys.append(key)

            orig_oracle_keys_ids = [x + "#" + str(sum([y == x for y in orig_oracle_keys[:i]])) for i, x in enumerate(orig_oracle_keys)]
            orig_oracle = sorted(orig_oracle_keys_ids, key=lambda x: correct_order_idxs[x])
           

            for i, key in enumerate(orig_oracle):
                key = key.split("#")[0]
                if key in ['name', 'developer', 'release_year', 'exp_release_date']:
                    val = ex['source']['mr_orig']['slots'][key]
                    if val != '':
                        val = 'PLACEHOLDER'
                    orig_oracle[i] = f'{key}={val}'
                elif key == 'specifier':
                    val = ex['source']['mr_orig']['slots']['specifier']
                    val = get_specifier_feats(val)
                    orig_oracle[i] = f'{key}={val}'
                elif '=' not in key:
                    orig_oracle[i] = f"{key}={ex['source']['mr_orig']['slots'][key]}"
                else:
                    orig_oracle[i] = key

            orig_oracle = (
                ex['source']['sequence']['oracle'][:2] + orig_oracle
            )
            ex['source']['sequence']['oracle_orig'] = orig_oracle
            orig_oracle_slots = orig_oracle[:2] + [x.split('=')[0] for x in orig_oracle[2:]]
            ex['source']['sequence']['oracle_slots_orig'] = orig_oracle_slots
            
            # Sanity check that we can get the original mr back from the sequence.
            test_orig_mr = {}
            for sf in set(orig_oracle[1:]):
                k,v = sf.split('=')
                if k in ['genres', 'player_perspective', 'platforms']:
                    if k not in test_orig_mr:
                        test_orig_mr[k] = []
                    test_orig_mr[k].append(v)
                    test_orig_mr[k] = sorted(test_orig_mr[k])
                elif v == 'N/A':
                    continue
                elif v == 'PLACEHOLDER' or k == 'specifier':
                    v = ex['source']['mr_orig']['slots'][k]
                    test_orig_mr[k] = v
                else:
                    test_orig_mr[k] = v
            assert test_orig_mr == ex['source']['mr_orig']['slots']

            print(json.dumps(ex), file=tmp_file)
        tmp_file.flush()
        shutil.copy2(tmp_file.name, str(path)) 

def generate_greedy_lm(sf_items, lm):
    inputs = list(sf_items)
    output = []
    t1 = '@'
    while inputs:
        scores = [lm[t1][t2] for t2 in inputs]
        idx = np.argmax(scores)
        t1 = inputs.pop(idx)
        output.append(t1)

    assert set(sf_items) == set(output)
    return output

def find_best_constrained_perm(sf_items, s_order, lm):

    platforms = [x for x in sf_items if x.startswith('platforms')]
    genres = [x for x in sf_items if x.startswith('genres')]
    perspectives = [x for x in sf_items 
                    if x.startswith('player_perspective')]
    
    m = {sf.split("=")[0]: sf.split('=')[1] for sf in sf_items
         if sf.split('=')[0] not in ['platforms', 'genres', 'player_perspective']}

    best_score = float('-inf')
    best_perm = None
    for platforms_p in permutations(platforms):
        for genres_p in permutations(genres):
            for perspectives_p in permutations(perspectives):
                platforms_l = list(platforms_p)
                genres_l = list(genres_p)
                perspectives_l = list(perspectives_p)
                p = {
                    'platforms': list(platforms_p),
                    'genres': list(genres_p),
                    'player_perspective': list(perspectives_p),
                }
                sf_tmp = list(s_order)
                sf_order = [
                    (x+"=" + m[x] if x in m else p[x].pop(0))
                    for x in sf_tmp
                ]
                score = lm_score(sf_order, lm)
                if score > best_score:
                    best_score = score
                    best_perm = sf_order
    return best_perm

def make_fixed_order_template(path):
    freqs = defaultdict(int)
    slot_counts = defaultdict(int)
    max_lens = defaultdict(int)
    for ex in example_iter(path):
        for slot in ex['source']['mr']['slots'].keys():
            slot_counts[slot] += 1
            if slot in ['genres', 'player_perspective', 'platforms']:
                max_lens[slot] = max(
                    max_lens[slot], len(ex['source']['mr']['slots'][slot]))
                for v in ex['source']['mr']['slots'][slot]:
                    freqs[f'{slot}={v}'] += 1
            elif slot == 'specifier':
                v = ex['source']['mr']['slots'][slot]
                freqs[f'{slot}={get_specifier_feats(v)}'] +=1
            else:
                v = ex['source']['mr']['slots'][slot]
                freqs[f'{slot}={v}'] += 1
                

    template = []
    for slot in sorted(slot_counts.keys(), key=lambda x: slot_counts[x]):
        if slot == 'rating':
            continue
        elif slot in max_lens:
            for i in range(1, max_lens[slot] + 1):
                template.append(f'{slot}=N/A-{i}')
        else:
            template.append(f'{slot}=N/A')

    def make_inc_order_fixed(mr):
        seq = list(template)
        for slot in mr['slots'].keys():
            if slot in ['genres', 'player_perspective', 'platforms']:
                vals = mr['slots'][slot]
                vals = sorted(vals, key=lambda x: freqs[f'{slot}={x}'])
                start_idx = seq.index(f'{slot}=N/A-1')
                for idx, val in enumerate(vals, start_idx):
                    seq[idx] = f'{slot}={val}'
                    
            elif slot == 'specifier':
                idx = seq.index(f'{slot}=N/A')
                seq[idx] = f'{slot}={get_specifier_feats(mr["slots"][slot])}'
                
            elif slot == 'rating':
                continue
            elif slot in ['name', 'developer', 'release_year', 
                          'exp_release_date']:
                v = mr['slots'][slot]
                if v != '':
                    v = 'PLACEHOLDER'
                idx = seq.index(f'{slot}=N/A')
                seq[idx] = f'{slot}={v}'
            else:
                idx = seq.index(f'{slot}=N/A')
                seq[idx] = f'{slot}={mr["slots"][slot]}'

        return [mr['da'], f"rating={mr['slots'].get('rating', 'N/A')}"] + seq

    def make_dec_order_fixed(mr):
        inc_order_fixed = make_inc_order_fixed(mr)
        dec_order_fixed = inc_order_fixed[:2] + inc_order_fixed[2:][::-1]
        return dec_order_fixed

    return make_inc_order_fixed, make_dec_order_fixed
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("valid", type=Path)
    parser.add_argument("test", type=Path)

    args = parser.parse_args()

    sf_map, sf_lm, sf_freqs = get_freq_mappings(args.train, 'oracle')
    s_map, s_lm, s_freqs = get_freq_mappings(args.train, 'oracle_slots')

    inc_order_fixed, dec_order_fixed = make_fixed_order_template(args.train)

    add_uncorrected_orderings(args.valid, sf_map, sf_lm, s_map, s_lm)
    add_uncorrected_orderings(args.test, sf_map, sf_lm, s_map, s_lm)

    add_orderings(args.valid, sf_map, sf_lm, sf_freqs, s_map, s_lm, 
                  inc_order_fixed, dec_order_fixed, orig=False)
    add_orderings(args.valid, sf_map, sf_lm, sf_freqs, s_map, s_lm,
                  inc_order_fixed, dec_order_fixed, orig=True)
    add_orderings(args.test, sf_map, sf_lm, sf_freqs, s_map, s_lm,
                  inc_order_fixed, dec_order_fixed, orig=False)
    add_orderings(args.test, sf_map, sf_lm, sf_freqs, s_map, s_lm,
                  inc_order_fixed, dec_order_fixed, orig=True)
    add_orderings(args.train, sf_map, sf_lm, sf_freqs, s_map, s_lm,
                  inc_order_fixed, dec_order_fixed, orig=False)

if __name__ == '__main__':
    main() 
