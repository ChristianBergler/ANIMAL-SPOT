"""
Module: checkpoints.py
Authors: Christian Bergler, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""

"""Code parts and implementations according to https://github.com/pytorch/tnt/blob/7b1dc6c/torchnet/meter/aucmeter.py

See also aucmeter.py

The AUCMeter measures the area under the receiver-operating characteristic
(ROC) curve for binary classification problems. The area under the curve (AUC)
can be interpreted as the probability that, given a randomly selected positive
example and a randomly selected negative example, the positive example is
assigned a higher score by the classification model than the negative example.

The AUCMeter is designed to operate on one-dimensional Tensors `output`
and `target`, where (1) the `output` contains model output scores that ought to
be higher when the model is more convinced that the example should be positively
labeled, and smaller when the model believes the example should be negatively
labeled (for instance, the output of a signoid function); and (2) the `target`
contains only values 0 (for negative examples) and 1 (for positive examples).

Code from https://github.com/pytorch/tnt/blob/7b1dc6c/torchnet/meter/aucmeter.py

BSD 3-Clause License

Copyright (c) 2017- Sergey Zagoruyko,
Copyright (c) 2017- Sasank Chilamkurthy, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Access Data: 12.09.2018, Last Access Date: 21.12.2021
"""

import numbers
import numpy as np
import torch


class ConfusionMeter:
    """Similar to AUCMeter"""

    def __init__(self, n_categories):
        self.n_categories = n_categories
        self.reset()

    def reset(self):
        self.confusion = torch.zeros(self.n_categories, self.n_categories)

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        assert np.ndim(output) == 1, "wrong output size (1D expected)"
        assert np.ndim(target) == 1, "wrong target size (1D expected)"
        assert (
            output.shape[0] == target.shape[0]
        ), "number of outputs and targets does not match"

        for o, t in zip(output, target):
            self.confusion[int(t)][int(o)] += 1

    def value(self):
        norm_confusion = self.confusion.clone()
        for i in range(self.n_categories):
            norm_f = norm_confusion[i].sum()
            if norm_f == 0:
                norm_f = 1
            norm_confusion[i] = norm_confusion[i] / norm_f
        return norm_confusion
