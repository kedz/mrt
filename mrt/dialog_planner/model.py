from plum2.object2 import Builder, LazyObject, no_init
from plum2.models.model import Model
import plum2.layer as layer
from plum2.search import GreedySearch, BeamSearch
from mrt.dialog_planner.constrained_beam_search import ConstrainedBeamSearch
from contextlib import contextmanager

from plum2.models import MODELS

@contextmanager
def dialog_planner(name):
    b = Seq2SeqModelBuilder()
    yield b
    m = b.finish()
    MODELS[name] = m
    b.forward()



class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, encoder_inputs=None,
                 decoder_input=None):
        super(EncoderDecoder, self).__init__()
        if encoder_inputs is None:
            encoder_inputs = ["encoder_input"]
        if decoder_input is None:
            decoder_input = "decoder_input"
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_inputs = encoder_inputs
        self.decoder_input = decoder_input
        self.searches = {}

    def __getstate__(self):
        object.__getattribute__(self, "init_deps")()
        state = {
            key: (val if key not in ["_dependencies", "searches"] else {})
            for key, val in object.__getattribute__(self, "__dict__").items()
        }
        return state

    @no_init
    def add_search(self, name, search):
        self.searches[name] = search

    def encode(self, batch):
        encoder_args = [batch[input] for input in self.encoder_inputs]

        state = self.encoder(*encoder_args)
        state['cons'] = encoder_args[0]
        return state

    def forward(self, batch):
        encoder_state = self.encode(batch)
        decoder_state = self.decoder(batch[self.decoder_input], encoder_state)
        decoder_state["encoder_state"] = encoder_state
        if len(self.searches):
            decoder_state["search"] = {
                search: LazyObject(lambda: search_func(encoder_state))
                for search, search_func in self.searches.items()
            }
            #print(list(decoder_state["search"].keys()))
        return decoder_state


class Seq2SeqModelBuilder(Builder):

    def __init__(self):
        super(Seq2SeqModelBuilder, self).__init__()
        self._arch = None
        self._layers = 1
        self._hidden_size = None
        self._emb_size = None
        self._enc_vocab = None
        self._dec_vocab = None
        self._bidir = False
        self._attention = None
        self._dropout = 0.0
        self._bridge = None
        self._searches = None
        self._tie_decoder_embeddings = False
    
    def arch(self, val):
        if val not in ["rnn", "gru", "lstm", "transformer"]:
            raise Exception("Invalid architecture.")
        self._arch = val
        return self

    def tie_decoder_embeddings(self, val=True):
        self._tie_decoder_embeddings = val
        return self

    def emb(self, dims):
        self._emb_size = dims
        return self
 
    def hidden(self, dims):
        self._hidden_size = dims
        return self
    
    def enc_vocab(self, vocab):
        self._enc_vocab = vocab
        return self

    def dec_vocab(self, vocab):
        self._dec_vocab = vocab
        return self
    
    def bi(self, val=True):
        self._bidir = val
        return self   

    def layers(self, val):
        self._layers = val
        return self

    def dropout(self, val):
        self._dropout = val
        return self

    def attn(self, val):
        if val not in layer.attention.TYPES or val == None:
            raise ValueError(f"Invalid attention method: {method} "
                             f"must be one of {layer.attention.TYPES}")
        self._attention = val
        return self

    def beam_search(self, beam_size=8, max_steps=100, name="beam"):
        if self._searches is None:
            self._searches = {}
        self._searches[name] = ConstrainedBeamSearch(
            self._dec_vocab, beam_size=beam_size, max_steps=max_steps)
        return self

    def greedy_search(self, max_steps=100, name="greedy"):
        if self._searches is None:
            self._searches = {}
        self._searches[name] = GreedySearch(self._dec_vocab, 
                                            max_steps=max_steps)
        return self
    
    def pass_thru_bridge(self):
        self._bridge = layer.seq.PassThruBridge()
        return self


    def _finish_object(self):
        
        if self._arch == 'transformer':
            
            enc_emb = layer.Embedding(
                self._enc_vocab.size(), self._emb_size, 
                pad_index=self._enc_vocab.pad_index, dropout=self._dropout)
            enc_pos_emb = layer.SinusoidalEmbedding(5000, self._emb_size,
                                                    dropout=self._dropout)
            enc_input_module = layer.ParallelModule(enc_emb, enc_pos_emb,
                                                    aggregate='sum')
            
            enc = layer.seq.TransformerEncoder(
                enc_input_module, hidden_feats=self._hidden_size, 
                layers=self._layers, dropout=self._dropout)

            dec_emb = layer.Embedding(
                self._dec_vocab.size(), self._emb_size,
                pad_index=self._dec_vocab.pad_index, dropout=self._dropout)

            dec_pos_emb = layer.SinusoidalEmbedding(5000, self._emb_size,
                                                    dropout=self._dropout)
            dec_input_module = layer.ParallelModule(dec_emb, dec_pos_emb,
                                                    aggregate='sum')

            predictor = layer.LinearPredictor(
                self._emb_size, self._dec_vocab.size())
 
            dec = layer.seq.TransformerDecoder(
                dec_input_module, self._hidden_size, predictor,
                layers=self._layers, dropout=self._dropout)

            model = EncoderDecoder(enc, dec)

            if self._searches:
                def make_search(model, s):
                    def f(encoder_state, controls=None):
                        s(model.decoder, encoder_state, controls=controls)
                        return s
                    return f

                for name, search in self._searches.items():
                    model.add_search(name, make_search(model, search))

        else:
            enc_emb = layer.Embedding(
                self._enc_vocab.size(), self._emb_size,
                pad_index=self._enc_vocab.pad_index, dropout=self._dropout)

            #if self._bidir and 
            if self._bridge == None:
                self._bridge = layer.seq.FeedForwardBridge(
                    self._hidden_size, self._bidir, self._arch, self._layers,
                    dropout=self._dropout)

            enc = layer.seq.RNNEncoder(enc_emb, self._hidden_size, self._arch,
                                       bidirectional=self._bidir,
                                       layers=self._layers, dropout=self._dropout,
                                       bridge=self._bridge)
                                       
            enc_out_feats = enc.out_feats
            dec_emb = layer.Embedding(
                self._dec_vocab.size(), self._emb_size,
                pad_index=self._dec_vocab.pad_index, dropout=self._dropout)

            if self._attention != None:
                attn = layer.attention.attention(enc_out_feats, self._hidden_size,
                                                 method=self._attention)
            else:
                attn = None

            adapter_feats = self._hidden_size + (attn != None) * enc.out_feats  

            predictor = layer.LinearPredictor(
                self._hidden_size, self._dec_vocab.size(),
                adapter_feats=adapter_feats, adapter_activation="tanh")

            dec = layer.seq.RNNDecoder(dec_emb, self._hidden_size, predictor,
                                       self._arch, layers=self._layers,
                                       dropout=self._dropout,
                                       attention_module=attn)
           
            model = EncoderDecoder(enc, dec)

            if self._searches:
                def make_search(model, s):
                    def f(encoder_state, controls=None):
                        s(model.decoder, encoder_state, controls=controls)
                        return s
                    return f

                for name, search in self._searches.items():
                    model.add_search(name, make_search(model, search))

        if self._tie_decoder_embeddings:
            Y = dec_emb.kernel 
            predictor._tie_embeddings = dec_emb

        return model
