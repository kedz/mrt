from plum2.object2 import Module


class PassThruPredictor(Module):
    def __init__(self):
        super(PassThruPredictor, self).__init__()

    def forward(self, inputs):
        return {'decoder_output': inputs}
