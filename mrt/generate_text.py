import torch
import json
from pathlib import Path

import plum2
from plum.types import Variable
from plum2.tasks.task import Task
from plum2.search import GreedySearch, BeamSearch

import mrt.eval_utils as eval_utils
import mrt.e2e.mr_utils
import mrt.e2e.rules
import mrt.viggo.mr_utils
import mrt.viggo.rules

import multiprocessing as mp



class GenerateText(Task):

    def __init__(self, model_path, rule_set, dataset_path, lin_strat, 
                 src_vocab, tgt_vocab, output_path,
                 orig_mr='false', beam_size=8):
        super(GenerateText, self).__init__()
        self.model_path = model_path
        self.rule_set = rule_set
        self.dataset_path = dataset_path
        self.orig_mr = orig_mr
        self.lin_strat = lin_strat
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.beam_size = beam_size
        self.output_path = output_path

    def input_iter(self):
        with open(str(self.dataset_path), 'r') as fp:
            for line in fp:
                yield line
                #ex = json.loads(line)
                #yield ex

    def get_reranked_index(self, results):
        results = list(results)
        
        results.sort(key=lambda x: x['err']['f1'], reverse=True)
        #results.sort(key=lambda x: x['mean_log_prob'], reverse=True)
        
        
        for i, result in enumerate(results):
            result['reranked_beam_candidate_num'] = i
            #print(f"{result['err']['f1']:0.4f}  {result['mean_log_prob']} {result['beam_candidate_num']}")

#        input()
        return results[0]['reranked_beam_candidate_num']        

    def make_batch(self, input_tokens, gpu):
        input_indices = [self.src_vocab.index(t) for t in input_tokens]
        input_indices = [self.src_vocab.start_index] \
            + input_indices + [self.src_vocab.stop_index]
        input_indices = torch.LongTensor([input_indices])
        lengths = torch.LongTensor([len(input_tokens) + 2])
        variable = Variable(input_indices.t(), lengths=lengths,
                            length_dim=0, batch_dim=1,
                            pad_value=self.src_vocab.pad_index).cuda(gpu)
        return {'encoder_input': variable}


   


    def run(self, env, verbose=False):
        self.env = env
        model = torch.load(str(self.model_path),
                           map_location=lambda storage, loc: storage)
        model.eval()
        if env['gpu'] > -1:
            model = model.cuda(env['gpu'])

        self.model = model

        output_path = Path(str(self.output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pool = mp.Pool(8)
        print(f"Writing: {output_path}")
        with output_path.open("w") as fp:
            for ex_id, ex in enumerate(pool.imap(self.process_example, self.input_iter())):
          #  for ex_id, line in enumerate(self.input_iter()):
                print(f"{ex_id}\r", end='', flush=True)

           #     ex = self.process_example(line)

                print(json.dumps(ex), file=fp, flush=True)
                    

    def process_example(self, line):
        if self.rule_set == 'E2E':
            rules = mrt.e2e.rules
            mr_utils = mrt.e2e.mr_utils
        elif self.rule_set == 'Viggo':
            rules = mrt.viggo.rules
            mr_utils = mrt.viggo.mr_utils
        else:
            raise Exception(f"Bad rule set: {self.rule_set}")


        model = self.model
        sch = BeamSearch(self.tgt_vocab, max_steps=100, 
                         beam_size=self.beam_size)

        use_orig = str(self.orig_mr) == 'true'


        ex = json.loads(line)
        if use_orig:
            mr = ex['orig']['mr']
        else:
            mr = ex['source']['mr']

        if str(self.lin_strat).startswith("random"):
            all_input_tokens = []
            for i in range(5):
                delex = str(self.lin_strat).endswith("delex")
                input_tokens = mr_utils.linearize_mr(
                    mr, delex=delex, order='random')
                all_input_tokens.append(input_tokens)
        else:
            all_input_tokens = [
                ex['source']['sequence'][str(self.lin_strat)]]

        results = []
        for input_tokens in all_input_tokens:

            batch = self.make_batch(input_tokens, self.env['gpu'])
            input_tokens_no_header = mr_utils.remove_header(
                input_tokens)
            input_tokens_no_header = [
                t for t in input_tokens_no_header
                if not t.endswith("N/A")
            ]
            
            encoder_state = model.encode(batch)
            sch(model.decoder, encoder_state)

            scores = sch._beam_scores[0].tolist()
            for i, output in enumerate(
                    sch.output(n_best=self.beam_size)[0]):
                pretty_output = mr_utils.detokenize(output)
                if str(self.lin_strat).endswith('delex'):
                    pretty_output = mr_utils.lexicalize_string(
                        pretty_output, **mr["slots"])
    

                mr_seq = rules.tag_tokens(output, **mr['slots'])
                lmr = mr_utils.tags2linear_mr(mr_seq, flip=True)
                prec, recall, fscore = eval_utils.lmr_prf(
                    input_tokens_no_header, lmr)
                results.append(
                    {
                        "beam_candidate_num": i,
                        "pretty": pretty_output, 
                        "tokens": output,
                        "tags": mr_seq,
                        "pred_lmr": lmr,
                        "mean_log_prob": scores[i],
                        "err": {
                            'precision': prec, 
                            'recall': recall, 
                            'f1': fscore},
                    })
        best_beam_cand = self.get_reranked_index(results)
        ex_out = {
            "mr": mr,
            "use_orig": use_orig,
            "lin_strat": str(self.lin_strat),
            "input": input_tokens,
            "outputs": results,
            "input_slot_fillers": input_tokens_no_header,
            "reranked_beam_output_index": best_beam_cand,
        }
        if 'target' in ex:
            ex_out['references'] = ex['target']['reference_strings']
        return ex_out
