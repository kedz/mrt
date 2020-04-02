from plum2.object2 import Module
import torch


class PointerDecoderWrapper(Module):
    def __init__(self, decoder):
        super(PointerDecoderWrapper, self).__init__()
        self.decoder = decoder

    
    def pointer_logits(self, decoder_state, encoder_state):

        decoder_output = decoder_state['decoder_output']\
            .permute_as_sequence_batch_features()

        encoder_output = encoder_state['output']\
            .permute_as_sequence_batch_features()
        logits = torch.einsum(
            'dbk,ebk->dbe', [decoder_output.data, encoder_output.data])
        logits = logits / (encoder_output.size(2) ** .5)
        logits = decoder_output.new_with_meta(logits)
        return logits

    def forward(self, inputs, encoder_state, prev_decoder_state=None,
                controls=None, search=False):

        decoder_state = self.decoder(
            inputs, encoder_state, prev_decoder_state, controls, search)
        target_logits = self.pointer_logits(decoder_state, encoder_state)
        decoder_state['target_logits'] = target_logits
        return decoder_state

    def next_state(self, prev_state, search_context, controls=None):
        next_state = self.decoder.next_state(
            prev_state, search_context, controls)
        logits = self.pointer_logits(
            next_state, {"output": search_context['encoder_output']})

        logits = logits.permute_as_sequence_batch_features()
        logits = logits.masked_fill(prev_state['pointer_mask'], float('-inf'))
        log_probs = logits.log_softmax(2)
        next_state['log_probs'] = log_probs
        return next_state
