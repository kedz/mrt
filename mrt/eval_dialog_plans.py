from plum2.tasks.task import Task
from plum2.tasks import TASKS
from permmr.pointer_search import GreedySearch, BeamSearch
from permmr.dialog_metric import DialogMetric
import torch
import json


def eval_dialog_plans(task_name, model_path, batches, vocab, beam=1):
    TASKS[task_name] = EvalDialogPlans(model_path, batches, vocab,
                                       beam_size=beam)

class EvalDialogPlans(Task):
    def __init__(self, model_path, batches, vocab, beam_size=1):
        super(EvalDialogPlans, self).__init__()
        self.model_path = model_path
        self.batches = batches
        self.vocab = vocab
        self.beam_size = beam_size

    def run(self, env, verbose=False):
        
        model = torch.load(self.model_path, 
                           map_location=lambda storage, loc: storage).eval()

        metric = DialogMetric(
            lambda m, b, fs: [[x for x in s if x != '<eos>'] 
                              for s in fs.output()],
            lambda m, b, fs: b['pretty_target'],
            lambda m, b, fs: b['da'])

        if env['gpu'] > -1:
            model = model.cuda(env['gpu'])
            self.batches.gpu = env['gpu']
        
        if self.beam_size > 1:
            search = BeamSearch(self.vocab, beam_size=self.beam_size, 
                                max_steps=20)
        else:
            search = GreedySearch(self.vocab, max_steps=20)

        with torch.no_grad():
            for step, batch in enumerate(self.batches, 1):
                print(f'{step}/{len(self.batches)}', end='\r', flush=True)
                encoder_state = model.encode(batch)

                fs = search(model.decoder, encoder_state)
                metric.update(model, batch, fs)
            print()
        print(metric.apply_metric())
            
             

