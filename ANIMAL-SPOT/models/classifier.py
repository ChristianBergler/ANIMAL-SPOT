"""
Module: classifier.py
Authors: Christian Bergler, Hendrik Schroeter
GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""

import torch
import torch.nn as nn

DefaultClassifierOpts = {
    "input_channels": 512,
    "pooling": "avg",  # avg, max
    "num_classes": 2,
}

"""
Defines classifier neural network consisting of an input, hidden, and output layer 
(fully connected classification part of the CNN) as well as the forward path computation.
"""
class Classifier(nn.Module):
    def __init__(self, opts: dict = DefaultClassifierOpts):
        super().__init__()
        self._opts = opts
        self._layer_output = dict()
        if opts["pooling"] == "avg":
            self.pooling = lambda x: torch.mean(x, dim=-1)
        elif opts["pooling"] == "max":
            self.pooling = lambda x: torch.max(x, dim=-1)[0]
        else:
            raise ValueError("Unkown pooling option")

        self.linear = nn.Linear(opts["input_channels"], opts["num_classes"])

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        hidden_layer = self.pooling(x)
        hidden_layer = hidden_layer.view(hidden_layer.size(0), -1)
        self._layer_output["hidden_layer_1"] = hidden_layer
        output_layer = self.linear(hidden_layer)
        self._layer_output["output_layer"] = output_layer
        return output_layer

    def model_opts(self):
        return self._opts

    def get_layer_output(self):
        return self._layer_output
