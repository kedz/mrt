import torch
import json

from plum.types import Variable
from plum2.tasks.task import Task
from plum2.search import GreedySearch, BeamSearch


class GenerateText(Task):

    def __init__(self, model_path, dataset, encoder_vocab, decoder_vocab,
                 order, output_prefix, search_widths=None):
        super(GenerateText, self).__init__()

        if search_widths == None:
            search_widths = [1, 8]
        self.model_path = model_path
        self.dataset = dataset
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        self.order = order
        self.search_widths = search_widths
        self.output_prefix = output_prefix

    def make_batch(self, input_tokens, gpu):
        input_indices = [self.encoder_vocab.index(t) for t in input_tokens]
        input_indices = torch.LongTensor([input_indices])
        lengths = torch.LongTensor([len(input_tokens)])

        variable = Variable(input_indices.t(), lengths=lengths,
                            length_dim=0, batch_dim=1,
                            pad_value=self.encoder_vocab.pad_index).cuda(gpu)

            
        return {'encoder_input': variable}

    def run(self, env, verbose=False):
        output_path = env['proj_dir'] / (
            self.output_prefix + "." + str(self.order) + ".jsonl")

        output_path.parent.mkdir(exist_ok=True, parents=True)

        
        model = torch.load(self.model_path, 
                           map_location=lambda storage, loc: storage)
        model.eval()
        if env['gpu'] > -1:
            model = model.cuda(env['gpu'])

        searches = {}

        for w in self.search_widths:
            if w == 1:
                searches['greedy'] = GreedySearch(self.decoder_vocab,
                                                  max_steps=100)
            else:
                searches[f'beam{w}'] = BeamSearch(self.decoder_vocab,
                                                  max_steps=100,
                                                  beam_size=w)

        with output_path.open("w") as fp:
            for i, ex in enumerate(self.dataset, 1):
                print(f"{i}/{len(self.dataset)}", end='\r', flush=True)
                input_tokens = (
                    [self.encoder_vocab.start_token] 
                    + ex['source']['sequence'][self.order]
                    + [self.encoder_vocab.stop_token] 
                )
                batch = self.make_batch(input_tokens, env['gpu'])
                encoder_state = model.encode(batch)

                result = {
                    'mr': ex['source']['mr'],
                    'encoder_input': input_tokens,
                    'outputs': {},
                }

                for search_name, search in searches.items():
                    if search_name == 'greedy':
                        search(model.decoder, encoder_state)
                        result['outputs']['greedy'] = search.output()[0]
                    else:
                        k = int(search_name[4:])
                        search(model.decoder, encoder_state)
                        result['outputs'][search_name] = search.output(
                            n_best=k)[0]
                print(json.dumps(result), file=fp)
            print()
