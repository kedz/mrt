from plum2.tasks.task import Task
from plum2.tasks import TASKS
from permmr.pointer_search import GreedySearch, BeamSearch
import torch
import json



def add_dialog_plans(task_name, model_path, dataset_path, pipeline, vocab):
    TASKS[task_name] = GenerateDialogPlans(model_path, dataset_path, pipeline,
                                           vocab)

class GenerateDialogPlans(Task):
    def __init__(self, model_path, dataset_path, pipeline, vocab):
        super(GenerateDialogPlans, self).__init__()
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.pipeline = pipeline
        self.vocab = vocab

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
 

    def run(self, env, verbose=False):

        model = torch.load(self.model_path, 
                           map_location=lambda storage, loc: storage).eval()
        if env['gpu'] > -1:
            model = model.cuda(env['gpu'])

        search = BeamSearch(self.vocab, beam_size=2, max_steps=20)
#        search = GreedySearch(self.vocab, max_steps=20)

        with open(self.dataset_path, 'r') as in_fp:
            batches = []
            for line in in_fp:
                example = json.loads(line)
                batches.append(example)
                if len(batches) == 4:
                #batch = self.pipeline([example])
                    batch = self.pipeline(batches)
                    if env['gpu'] > -1:
                        batch = self.batch2gpu(batch, env['gpu'])
                    encoder_state = model.encode(batch)
                    encoder_state['source_indices'] = batch['encoder_input']
                    search(model.decoder, encoder_state)
            
                    plans = search.output() #(n_best=4)
                    for b, pred_plan in enumerate(plans):
                        pred_plan = pred_plan[:-1]
                        #pr
                        print(pred_plan)
                        ref_plan = batches[b]['source']['sequence']['random'][2:]
                        print(ref_plan)
                        #print(plan[1][:-1])
                        #print(plan[2][:-1])
                        #print(plan[3][:-1])
                        print()

                    #print(plan)
                        assert len(pred_plan) == len(ref_plan)
                        assert set(ref_plan) == set(pred_plan)
            #print(batches[1]['source']['sequence']['oracle'][2:])
            #print(batches[2]['source']['sequence']['oracle'][2:])
            #assert len(plan) == len(set(example['source']['sequence']['random'][2:]))
                    batches = []
