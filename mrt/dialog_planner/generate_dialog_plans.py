from plum.types import Variable
from plum2.tasks.task import Task
from plum2.tasks import TASKS
from plum2.search import GreedySearch, BeamSearch
from mrt.dialog_planner.constrained_beam_search import ConstrainedBeamSearch
import torch
import numpy as np
import json
from tempfile import NamedTemporaryFile
import shutil



def add_dialog_plans(task_name, model_path, dataset_path, vocab, mr_utils, lin_strat):
    TASKS[task_name] = GenerateDialogPlans(model_path, dataset_path, vocab,
                                           mr_utils, lin_strat)

class GenerateDialogPlans(Task):
    def __init__(self, model_path, dataset_path, vocab, mr_utils, lin_strat):
        super(GenerateDialogPlans, self).__init__()
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.vocab = vocab
        self.mr_utils = mr_utils
        self.lin_strat = lin_strat

    def batch2gpu(self, batch, gpu):
        if isinstance(batch, dict):
            for key, value in batch.items():
                batch[key] = self.batch2gpu(value, gpu)
            return batch
        elif isinstance(batch, list):
            return [self.batch2gpu(item, gpu) for item in batch] 
        elif isinstance(batch, tuple):
            return tuple([self.batch2gpu(item, gpu) for item in batch])
        elif hasattr(batch, "cuda"):
            return batch.cuda(gpu)
        else:
            return batch


    def make_batch(self, seq, gpu):
        tokens = [self.vocab.start_token] + seq + [self.vocab.stop_token]
        idxs = torch.LongTensor([[self.vocab.index(t)] for t in tokens])
        lengths = torch.LongTensor([len(tokens)])
        encoder_inputs = Variable(
            idxs, lengths, batch_dim=1, length_dim=0, pad_value=0)
        return {'encoder_input': encoder_inputs.cuda(gpu)}

    def process(self, example, model, gpu):
        ref_lmr_wheader = example['source']['sequence'][f'{self.lin_strat}_delex']
        #ref_lmr_wheader = self.mr_utils.linearize_mr(
        #    example['source']['mr'], delex=True)
        header = self.mr_utils.get_header(ref_lmr_wheader)
        ref_lmr = self.mr_utils.remove_header(ref_lmr_wheader)

#        print(example['source']['sequence']['inc_freq_fixed_delex']) 
        batch = self.make_batch(
            example['source']['sequence'][f'{self.lin_strat}_delex'],
            gpu) 
        encoder_state = model.encode(batch)
        search = ConstrainedBeamSearch(self.vocab, beam_size=32, max_steps=20)

        search(model.decoder, encoder_state)
        output = search.output()[0][:-1]

        if len(output) < len(ref_lmr):
            print("BADNESS!")
            output = output + [x for x in ref_lmr if x not in output]

        example['source']['sequence']['nlm_delex'] = header + output
        assert set(example['source']['sequence']['nlm_delex']) \
            == set(example['source']['sequence'][f'{self.lin_strat}_delex'])

        
        slots = {x.split("=")[0]: x.split("=")[1] 
                 for x in self.mr_utils.remove_header(
                     example['source']['sequence'][f'{self.lin_strat}_lex'])}
        output_lex = []
        for t in output:
            if t.endswith("PLACEHOLDER"):
                slot = t.split("=")[0]
                output_lex.append(f'{slot}={slots[slot]}')
            elif "SPECIFIER" in t:
                slot = t.split("=")[0]
                output_lex.append(f'{slot}={slots[slot]}')
            else:
                output_lex.append(t)
        example['source']['sequence']['nlm_lex'] = header + output_lex
        assert set(example['source']['sequence']['nlm_lex']) \
            == set(example['source']['sequence'][f'{self.lin_strat}_lex'])

    def run(self, env, verbose=False):

        model = torch.load(str(self.model_path), 
                           map_location=lambda storage, loc: storage).eval()
        if env['gpu'] > -1:
            model = model.cuda(env['gpu'])

        with NamedTemporaryFile('w') as out_fp:
            with open(str(self.dataset_path), 'r') as in_fp:
                for i, line in enumerate(in_fp, 1):
                    print(f'processing: {i}', end='\r', flush=True)
                    example = json.loads(line)
                    self.process(example, model, env['gpu'])
                    print(json.dumps(example), file=out_fp)
                print()
            out_fp.flush()
            shutil.copy(out_fp.name, str(self.dataset_path))
