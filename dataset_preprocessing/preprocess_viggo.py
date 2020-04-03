import argparse
import csv
import json
from pathlib import Path
import re
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from mrt.viggo.tagging_rules import (
    tag_slots, tags2mrseq, slots2mr, get_specifier_feats)

CATEGORICAL = ["name", "developer", "release_year", "exp_release_date", 
               "esrb", "has_multiplayer", "available_on_steam",
               "has_linux_release", "has_mac_release", "rating", "specifier"]
LIST = ["genres", "player_perspective", "platforms"]

DELEX_TOKENS = ['NAME', 'DEVELOPER', 'RELEASE_YEAR', 'EXP_RELEASE_DATE',
                'SPECIFIER']

VA_CORRECT = {
    13: ['developer=PLACEHOLDER', 'genres=sport', 'platforms=Xbox', 'name=PLACEHOLDER'],    
    103: ['has_linux_release=no', 'name=PLACEHOLDER', 'has_mac_release=yes', 'platforms=PC', 'available_on_steam=yes', 'genres=real-time strategy'],
    167: ['name=PLACEHOLDER', 'player_perspective=bird view', 'genres=real-time strategy', 'developer=PLACEHOLDER', 'genres=adventure', 'release_year=PLACEHOLDER'],
    250: ['genres=role-playing', 'genres=action-adventure', 'player_perspective=third person', 'genres=puzzle', 'release_year=PLACEHOLDER', 'genres=adventure', 'player_perspective=bird view', 'name=PLACEHOLDER', 'developer=PLACEHOLDER', 'platforms=Nintendo'],
    356: ['has_linux_release=yes', 'name=PLACEHOLDER', 'esrb=M (for Mature)'],
    413: ['player_perspective=first person', 'developer=PLACEHOLDER', 'genres=puzzle', 'name=PLACEHOLDER'],
}

TR_CORRECT = {
    71: ['has_mac_release=no', 'genres=simulation', 'genres=strategy', 'name=PLACEHOLDER', 'has_multiplayer=no'],
    181: ['name=PLACEHOLDER', 'developer=PLACEHOLDER', 'genres=driving/racing'],
    425: ['name=PLACEHOLDER', 'has_multiplayer=yes', 'genres=action', 'genres=shooter', 'platforms=PlayStation'],
    522: ['developer=PLACEHOLDER', 'platforms=PC', 'name=PLACEHOLDER', 'has_mac_release=no'],
    824: ['name=PLACEHOLDER', 'genres=role-playing', 'genres=MMORPG', 'player_perspective=first person', 'platforms=PC'],
    1403: ['has_linux_release=yes', 'has_multiplayer=no', 'name=PLACEHOLDER'],
    1580: ['player_perspective=first person', 'genres=action', 'name=PLACEHOLDER', 'genres=shooter'],
    1585: ['developer=PLACEHOLDER', 'genres=tactical', 'name=PLACEHOLDER', 'player_perspective=first person', 'genres=role-playing', 'genres=shooter'],
    1644: ['release_year=PLACEHOLDER', 'genres=shooter', 'player_perspective=third person', 'genres=adventure', 'name=PLACEHOLDER', 'platforms=PlayStation', 'genres=role-playing', 'genres=action'],
    1739: ['esrb=E (for Everyone)', 'release_year=PLACEHOLDER', 'genres=driving/racing', 'name=PLACEHOLDER', 'has_mac_release=no', 'has_multiplayer=no', 'platforms=PlayStation', 'platforms=Xbox', 'has_linux_release=yes', 'genres=sport', 'genres=simulation', 'platforms=PC'],
    1744: ['player_perspective=bird view', 'has_multiplayer=yes'],
    1750: ['has_mac_release=yes', 'platforms=Nintendo', 'platforms=PC', 'available_on_steam=yes', 'has_multiplayer=no', 'genres=arcade', 'genres=strategy', 'has_linux_release=yes', 'genres=puzzle', 'name=PLACEHOLDER'],
    1830: ['genres=action-adventure', 'genres=shooter', 'player_perspective=first person', 'developer=PLACEHOLDER', 'esrb=M (for Mature)', 'release_year=PLACEHOLDER', 'name=PLACEHOLDER'],
    1896: ['name=PLACEHOLDER', 'genres=simulation', 'genres=turn-based strategy', 'esrb=E 10+ (for Everyone 10 and Older)', 'has_multiplayer=yes', 'platforms=PC'],
    1941: ['name=PLACEHOLDER', 'genres=driving/racing', 'developer=PLACEHOLDER', 'release_year=PLACEHOLDER'],
    2041: ['genres=action-adventure', 'genres=fighting', 'developer=PLACEHOLDER', 'genres=platformer', 'esrb=T (for Teen)', 'player_perspective=first person', 'release_year=PLACEHOLDER', 'name=PLACEHOLDER'],
    2068: ['name=PLACEHOLDER', 'developer=PLACEHOLDER', 'platforms=Nintendo', 'esrb=E (for Everyone)'],
    2276: ['genres=sport', 'genres=vehicular combat', 'has_multiplayer=yes', 'name=PLACEHOLDER'],
    2282: ['esrb=E (for Everyone)', 'genres=turn-based strategy', 'player_perspective=bird view', 'name=PLACEHOLDER', 'genres=strategy', 'has_multiplayer=yes'],
    2323: ['name=PLACEHOLDER', 'developer=PLACEHOLDER', 'has_multiplayer=yes', 'genres=strategy'],
    2486: ['genres=shooter', 'player_perspective=third person', 'release_year=PLACEHOLDER', 'name=PLACEHOLDER'],
    2748: ['has_linux_release=no', 'genres=adventure', 'name=PLACEHOLDER', 'genres=puzzle', 'platforms=PC', 'available_on_steam=yes', 'genres=platformer', 'has_mac_release=no', 'platforms=PlayStation', 'platforms=Xbox'],
    2853: ['name=PLACEHOLDER', 'genres=puzzle', 'player_perspective=first person', 'platforms=Xbox', 'genres=shooter', 'release_year=PLACEHOLDER', 'platforms=PlayStation', 'genres=platformer', 'platforms=PC'],
    3183: ['name=PLACEHOLDER', 'developer=PLACEHOLDER', 'genres=fighting', 'player_perspective=first person', 'genres=action-adventure'],
    3256: ['release_year=PLACEHOLDER', 'name=PLACEHOLDER', 'has_mac_release=yes'],
    3417: ['player_perspective=bird view', 'has_mac_release=no', 'has_linux_release=no', 'esrb=T (for Teen)', 'platforms=PC', 'genres=real-time strategy', 'name=PLACEHOLDER'],
    3490: ['platforms=PlayStation', 'name=PLACEHOLDER', 'release_year=PLACEHOLDER', 'has_mac_release=yes', 'platforms=PC', 'platforms=Nintendo', 'has_linux_release=no', 'genres=simulation', 'genres=sport', 'available_on_steam=yes', 'platforms=Xbox'],
    3895: ['developer=PLACEHOLDER', 'genres=shooter', 'name=PLACEHOLDER', 'player_perspective=first person'],
    4184: ['name=PLACEHOLDER', 'release_year=PLACEHOLDER', 'genres=sport'],
    4337: ['platforms=Nintendo', 'platforms=PlayStation', 'genres=sport', 'platforms=PC', 'available_on_steam=no', 'platforms=Xbox', 'has_mac_release=yes', 'name=PLACEHOLDER', 'has_linux_release=yes'],
    4558: ['developer=PLACEHOLDER', 'has_linux_release=no', 'available_on_steam=yes', 'platforms=Nintendo', 'name=PLACEHOLDER', 'platforms=PlayStation', 'platforms=PC', 'platforms=Xbox', 'has_mac_release=yes', 'genres=sport'],
    4648: ['platforms=PlayStation', 'genres=sport', 'genres=vehicular combat', 'name=PLACEHOLDER'],
    4891: ['available_on_steam=yes', 'name=PLACEHOLDER', 'has_linux_release=yes', 'esrb=M (for Mature)'],
    4943: ['has_linux_release=yes', 'genres=strategy', 'has_mac_release=yes', 'name=PLACEHOLDER'],
    5062: ['name=PLACEHOLDER', 'platforms=Xbox', 'genres=sport'],
}

TE_CORRECT = {
    #22: ['genres=shooter', 'name=PLACEHOLDER', 'has_mac_release=yes', 'genres=role-playing', 'genres=tactical'],
    52: ['genres=action', 'player_perspective=first person', 'has_multiplayer=yes', 'release_year=PLACEHOLDER', 'genres=shooter', 'name=PLACEHOLDER', 'developer=PLACEHOLDER'],
    233: ['release_year=PLACEHOLDER', 'available_on_steam=no', 'genres=vehicular combat', 'name=PLACEHOLDER'],
    235: ['developer=PLACEHOLDER', 'platforms=PlayStation', 'genres=action-adventure', 'name=PLACEHOLDER', 'platforms=PC', 'esrb=M (for Mature)', 'genres=role-playing', 'platforms=Xbox'],
    480: ['release_year=PLACEHOLDER', 'platforms=PlayStation', 'genres=driving/racing', 'player_perspective=third person', 'developer=PLACEHOLDER', 'genres=vehicular combat', 'name=PLACEHOLDER'],
    606: ['developer=PLACEHOLDER', 'player_perspective=side view', 'genres=puzzle', 'name=PLACEHOLDER'],
    875: ['player_perspective=first person', 'platforms=PlayStation', 'genres=shooter', 'platforms=Xbox', 'has_linux_release=no', 'has_mac_release=no', 'platforms=Nintendo', 'platforms=Nintendo Switch', 'genres=role-playing', 'name=PLACEHOLDER', 'available_on_steam=yes', 'platforms=PC'],

}

def preprocess_train(path, output):
    global COUNTS
    COUNTS = 0
    with path.open('r') as in_fp, output.open("w") as out_fp:
        reader = csv.reader(in_fp)
        next(reader)

        for mr_line, utt_line in reader:
            example = preprocess_example(mr_line, utt_line, TR_CORRECT)
            print(json.dumps(example), file=out_fp)

def preprocess_valid(path, output):
    global COUNTS
    COUNTS = 0
    with path.open('r') as in_fp, output.open("w") as out_fp:
        reader = csv.reader(in_fp)
        next(reader)

        for mr_line, utt_line in reader:
            example = preprocess_example(mr_line, utt_line, VA_CORRECT)
            print(json.dumps(example), file=out_fp)

def preprocess_test(path, output):
    global COUNTS
    COUNTS = 0
    with path.open('r') as in_fp, output.open("w") as out_fp:
        reader = csv.reader(in_fp)
        next(reader)

        for mr_line, utt_line in reader:
            example = preprocess_example(mr_line, utt_line, TE_CORRECT)
            print(json.dumps(example), file=out_fp)


COUNTS = 0
SPECS = set()
import copy
def preprocess_example(mr_line, utt_line, CORRECT):

    mr = extract_mr(mr_line)
    orig_mr = copy.copy(mr)
    orig_utt = utt_line
    reference_text, tokens, tokens_delex = extract_utterance(utt_line, mr) 

    if 'specifier' in mr['slots']:
        spec = mr['slots']['specifier']
        sup_feat = 'S' if 'worst' in spec or spec[-2:] == 'st' else 'R'
        start_feat = 'V' if spec[0] in 'aeiou' else 'C'
        idx = tokens_delex.index('SPECIFIER')
        tokens_delex[idx] = f'SPECIFIER_{start_feat}_{sup_feat}'
        spec_placeholder = f'SPECIFIER_{start_feat}_{sup_feat}'

    tags = tag_slots(tokens_delex)

    for tok, tag in zip(tokens_delex, tags):
        print(f'{tag:40s} {tok}')
    print()

    pred_mrseq = tags2mrseq(tags)
    pred_mr = set(pred_mrseq)

    gt_mr = slots2mr(mr['slots'])
    gt_mr = set([x for x in gt_mr if not x.startswith('rating')])
    #print(gt_mr)
    #oracle_orig = ['0'] * len(gt_mr
    #input()

    global COUNTS
      
    COUNTS+=1  
    if COUNTS in CORRECT:
        gt_mr = set(CORRECT[COUNTS])

    print(gt_mr)
    print(pred_mr)


    print(COUNTS)
    if COUNTS >= 0:
        if gt_mr != pred_mr:
            input()
    
    oracle_seq = [
        mr['da'], f"rating={mr['slots'].get('rating', 'N/A')}"
    ] + pred_mrseq
    oracle_slots = oracle_seq[:2] + [x.split('=')[0] for x in oracle_seq[2:]]
    #oracle_slots = [x+f"_{sum([x==y for y in oracle_slots[:i]])}" for i, x in enumerate(oracle_slots)]

    rating = mr['slots'].get('rating', None)
    new_mr = {"da": mr["da"], "slots": {}}
    if rating:
        new_mr['slots']['rating'] = rating
    
    for sf in pred_mrseq:
        s, f = sf.split('=')
        if s in ['platforms', 'player_perspective', 'genres']:
            if s not in new_mr['slots']:
                new_mr['slots'][s] = []
            new_mr['slots'][s].append(f)
        elif s == 'specifier':
            new_mr['slots']['specifier'] = mr['slots']['specifier']
        else:
            if f == 'PLACEHOLDER':
                f = mr['slots'][s]
            new_mr['slots'][s] = f
    
    for s, f in new_mr['slots'].items():
        if isinstance(f, list):
            new_mr['slots'][s] = sorted(f)



    example = {
        "original": {
            "mr": mr_line, 
            "utt": utt_line,
        },
        "source": {
            "mr_orig": orig_mr,
            "mr": new_mr,
            'sequence': {
                'oracle': oracle_seq,
                'oracle_slots': oracle_slots,
            },
        },
        "target": {
            "sequence": {
                "delex": tokens_delex,
                "lex": tokens,
            },
            "reference_string": reference_text,
        },
    }
    
    from pprint import pprint
    pprint(example)
#    print(pred_mrseq)
#    print(oracle_seq)
#    #input()
#    ff COUNTS in CORRECT:
#            input()
    return example

def extract_mr(line):
    m = re.match(r'^(.*?)\((.*)\)$', line)
    da = m.group(1)
    da_args = m.group(2)
    mr = {"da": da, "slots": {}}
    for slot, vals in re.findall(r'([\w_]+)\[(.*?)\]', da_args):

        if slot in CATEGORICAL:
            mr["slots"][slot] = vals
        elif slot in LIST:
            mr["slots"][slot] = sorted(vals.split(", "))
        if slot == "specifier":
            mr["slots"]["specifier"] = vals.replace(" ", "_")

    return mr

def extract_utterance(ref, mr):
    ref = re.sub(r"Speed:Payback", r"Speed: Payback", ref)
    ref = re.sub(r"Evolution Studios'", "Evolution Studios '", ref)
    ref = re.sub(r"'09", r"2009", ref)
    ref = re.sub(r"  ", r" ", ref)

    delex_ref = str(ref)
    
    if "developer" in mr["slots"] and mr["slots"]["developer"] != "":
        assert mr["slots"]["developer"].lower() in delex_ref.lower()
        delex_ref = re.sub(
            re.escape(mr["slots"]["developer"]), 
            "DEVELOPER", 
            delex_ref,
            flags=re.I)

    if "name" in mr["slots"] and mr["slots"]["name"] != "":
        assert mr["slots"]["name"].lower() in delex_ref.lower()
        delex_ref = re.sub(
            re.escape(mr["slots"]["name"]),
            "NAME",
            delex_ref, 
            flags=re.I)

    if "exp_release_date" in mr["slots"] \
            and mr["slots"]["exp_release_date"] != "":
        assert mr["slots"]["exp_release_date"].lower() in delex_ref.lower()
        delex_ref = re.sub(
            re.escape(mr["slots"]["exp_release_date"]),
            "EXP_RELEASE_DATE",
            delex_ref,
            flags=re.I)
 
    if "release_year" in mr["slots"] and mr["slots"]["release_year"] != "":
        assert mr["slots"]["release_year"].lower() in delex_ref.lower()
        delex_ref = re.sub(
            re.escape(mr["slots"]["release_year"]),
            "RELEASE_YEAR",
            delex_ref,
            flags=re.I)

    if "specifier" in mr["slots"] and mr["slots"]["specifier"] != '':
        spec = mr["slots"]["specifier"]
        if "_" in spec:
            delex_ref = re.sub(spec.replace("_", " "), spec, delex_ref)
        if spec == "fast-paced":
            delex_ref = re.sub("fast paced", spec, delex_ref)
            
        assert spec in delex_ref
        delex_ref = re.sub(spec, "SPECIFIER", delex_ref)
    
    delex_ref = re.sub(r"fps\.", r"fp__ __s.", delex_ref)
    delex_ref = re.sub(r"fps", r"fp__ __s", delex_ref)
    delex_ref = re.sub(r"FPS\.", r"FP__ __S.", delex_ref)
    delex_ref = re.sub(r"FPS", r"FP__ __S", delex_ref)
    delex_ref = re.sub(r"(\w)/(\w)", r"\1 / \2", delex_ref)
    delex_ref = re.sub(r"(\w)-(\w)", r"\1 - \2", delex_ref)
    if '/' in delex_ref or '-' in delex_ref or "__" in delex_ref:
        print(delex_ref)
        #input()

    delex_sents = sent_tokenize(delex_ref)
    delex_tokens = " <sent> ".join(
        [" ".join(word_tokenize(delex_sent)) 
         for delex_sent in delex_sents]).split()

    for i, t in enumerate(delex_tokens):
        if t not in DELEX_TOKENS:
            delex_tokens[i] = t.lower()

    x = " ".join(delex_tokens)
    x = x.replace("adventure-shooters", "adventure - shooters")
    x = x.replace('top-down', 'top - down')
    x = x.replace('third-', 'third -')
    x = x.replace('-person', '- person')
    x = x.replace('puzzle-based', 'puzzle - based')
    x = x.replace("puzzle-oriented", 'puzzle - oriented')
    x = x.replace("adventure-filled", 'adventure - filled')
    x = x.replace("non-availability", "non - availability")
    x = x.replace("bird's-eye", "bird 's - eye")
    x = x.replace("n't not", "not")
    x = x.replace('release_year-released', 'RELEASE_YEAR released')
    x = x.replace('puzzle-platforming','puzzle platforming')
    x = x.replace('shooter-rpg','shooter - rpg')
    x = x.replace("\\name", "NAME")
    x = x.replace('m- rated', 'm - rated')
    x = x.replace('m-rated', 'm - rated')
    x = x.replace('bird- view', 'bird - view')
    x = x.replace('action- adventure', 'action-adventure')
    x = x.replace("role-playing-text", "role-playing - text")
    x = x.replace(" m. ", " m . <sent> ")
    x = x.replace(" t. ", " t . <sent> ")
    x = x.replace(" e. ", " e . <sent> ")
    x = x.replace('third-person', 'third - person')
    x = x.replace('strategy-simulation', 'strategy - simulation')
    x = x.replace('pc-only', 'pc - only')
    x = x.replace('bird- and', 'bird and')
    x = x.replace('music-game', 'music - game')
    x = x.replace("simulation-sports", "simulation - sports")
    x = x.replace('single-player-only', 'single-player - only')
    x = x.replace('action-platformers', 'action - platformers')
    x = x.replace('nfs : payback', 'NAME')
    x = x.replace('simulaton', 'simulation')

    delex_tokens = x.split()

    lex_tokens = list(delex_tokens)
    for slot in DELEX_TOKENS:
        while slot in lex_tokens:
            idx = lex_tokens.index(slot)
            filler_toks = word_tokenize(
                mr['slots'][slot.lower()].replace('_', ' '))
            repl = [tok.lower() for tok in filler_toks]
            lex_tokens = lex_tokens[:idx] + repl + lex_tokens[idx+1:]
            
    lex_ref = delex_ref
    for slot in DELEX_TOKENS:
        if slot.lower() in mr['slots']:
            lex_ref = re.sub(slot, mr['slots'][slot.lower()], lex_ref)

    lex_ref = re.sub(" / ", "/", lex_ref)
    lex_ref = re.sub(" - ", "-", lex_ref)
    lex_ref = re.sub("__ __", "", lex_ref)
     
    return lex_ref, lex_tokens, delex_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("valid", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.mkdir(exist_ok=True, parents=True)
    preprocess_train(args.train, args.output / "viggo.train.jsonl")
    preprocess_valid(args.valid, args.output / "viggo.valid.jsonl")
    preprocess_test(args.test, args.output / "viggo.test.jsonl")

if __name__ == '__main__':
    main()
