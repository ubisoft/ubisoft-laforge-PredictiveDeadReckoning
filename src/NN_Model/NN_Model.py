import numpy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
from collections import OrderedDict
import datetime


class NeuralNetworkModel(torch.nn.Module):
    def __init__(self, num_inputs=13, num_outputs=4, hidden_layer_size=100, activation_func="RelU", using_gpu=0):
        super().__init__()

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._hidden_layer_size = hidden_layer_size
        self._activation_func = activation_func
        self.using_gpu = using_gpu

        self.init_model()

    def init_model(self):
        self.input_layer = torch.nn.Linear(self._num_inputs, self._hidden_layer_size, dtype=float)
        self.r0 = torch.nn.ReLU()

        self.hidden_layer_1 = torch.nn.Linear(self._hidden_layer_size, self._hidden_layer_size, dtype=float)
        self.r1 = torch.nn.ReLU()

        self.hidden_layer_2 = torch.nn.Linear(self._hidden_layer_size, self._hidden_layer_size, dtype=float)
        self.r2 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(self._hidden_layer_size, self._num_outputs, dtype=float)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.r0(x)

        x = self.hidden_layer_1(x)
        x = self.r1(x)

        x = self.hidden_layer_2(x)
        x = self.r2(x)

        return self.output_layer(x)
