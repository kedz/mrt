from plum2.object2 import Builder, LazyObject, no_init
from plum2.models.model import Model
import plum2.layer as layer
from plum2.models.seq2seq import EncoderDecoder
from mrt.dialog_planner.passthru_predictor import PassThruPredictor
from mrt.dialog_planner.pointer_decoder_wrapper import PointerDecoderWrapper
from mrt.dialog_planner.pointer_search import GreedySearch, BeamSearch
from contextlib import contextmanager


from plum2.models import MODELS

@contextmanager
def dialog_planner(name):
    b = DialogPlannerBuilder()
    yield b
    m = b.finish()
    MODELS[name] = m
    b.forward()


class DialogPlannerBuilder(Builder):

    def __init__(self):
        super(DialogPlannerBuilder, self).__init__()

        self._layers = 1
        self._dropout = 0.0
        self._emb_feats = None
        self._hidden_feats = None
        self._vocab = None
        self._searches = None
        self._tie_embs = False

    def tie_embs(self, val=True):
        self._tie_embs = val
        return self
    
    def layers(self, val):
        self._layers = val
        return self

    def emb(self, val):
        self._emb_feats = val
        return self

    def hidden(self, val):
        self._hidden_feats = val
        return self

    def dropout(self, val):
        self._dropout = val
        return self

    def vocab(self, val):
        self._vocab = val
        return self

    def beam_search(self, beam_size=4, max_steps=20, name="beam"):
        if self._searches is None:
            self._searches = {}
        self._searches[name] = BeamSearch(self._vocab, 
                                          beam_size=beam_size,
                                          max_steps=max_steps)
        return self



    def _finish_object(self):
        enc_emb = layer.Embedding(
            self._vocab.size(),
            self._emb_feats,
            pad_index=self._vocab.pad_index,
            dropout=self._dropout)
        enc = layer.seq.TransformerEncoder(
            enc_emb, hidden_feats=self._hidden_feats,
            layers=self._layers, dropout=self._dropout)

        dec_emb = layer.Embedding(
                self._vocab.size(),
                self._emb_feats,
                pad_index=self._vocab.pad_index, 
                dropout=self._dropout)

        dec_pos_emb = layer.SinusoidalEmbedding(
            5000, 
            self._emb_feats,
            dropout=self._dropout)
        dec_input_module = layer.ParallelModule(
            dec_emb, dec_pos_emb,
            aggregate='sum')

        predictor = PassThruPredictor()
        #predictor = layer.LinearPredictor(
        #    self._emb_feats, self._vocab.size())
         
        dec = PointerDecoderWrapper(
            layer.seq.TransformerDecoder(
                dec_input_module, self._hidden_feats, predictor,
                layers=self._layers, dropout=self._dropout))

        model = PointerEncoderDecoder(enc, dec)
        if self._searches:
            def make_search(model, s):
                def f(encoder_state, controls=None):
                    s(model.decoder, encoder_state, controls=controls)
                    return s
                return f

            for name, search in self._searches.items():
                model.add_search(name, make_search(model, search))

        if self._tie_embs:
            enc_emb.kernel = dec_emb.kernel
        

        return model


class PointerEncoderDecoder(EncoderDecoder):
    def __init__(self, *args, source_indices='encoder_input', **kwargs):
        super(PointerEncoderDecoder, self).__init__(*args, **kwargs)
        self.source_indices = source_indices

    def encode(self, batch):
        encoder_state = super(PointerEncoderDecoder, self).encode(batch)
        encoder_state['source_indices'] = batch[self.source_indices]
        return encoder_state

