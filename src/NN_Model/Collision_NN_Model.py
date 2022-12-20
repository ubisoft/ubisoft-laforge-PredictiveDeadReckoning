import numpy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
from collections import OrderedDict
import datetime

class NeuralNetworkModel:
    def __init__(self, num_hidden_layers = 2, num_inputs = 13, num_outputs = 4, hidden_layer_size = 100, activation_func = "RelU", using_gpu=0):
        self._model_dict = OrderedDict()
        self.train_data_x = []
        self.train_data_y = []
        self.test_data_x = []
        self.test_data_y = []
        self._model = 0
        self._num_hidden_layers = num_hidden_layers
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._hidden_layer_size = hidden_layer_size
        self._activation_func = activation_func
        self.using_gpu = using_gpu

    def init_model(self):
        self._model_dict['input_layer'] = torch.nn.Linear(self._num_inputs, self._hidden_layer_size)
        self._add_activation_function(ind='input')

        for _i in range(self._num_hidden_layers):
            self._model_dict['hidden_layer_' + str(_i+1)] = torch.nn.Linear(self._hidden_layer_size, self._hidden_layer_size)
            self._add_activation_function(ind=(_i + 1))

        self._model_dict['output_layer'] = torch.nn.Linear(self._hidden_layer_size, self._num_outputs)
        self._model = torch.nn.Sequential(self._model_dict)

    def _add_activation_function(self, ind=None):
        if (self._activation_func == "RelU"):
            self._model_dict['activation_' + str(ind)] = torch.nn.ReLU()

    def __str__(self):
        to_print = "\t======================================================================\n"
        to_print += "\tLayer Name" + "\t" +"Layer Type" + "\t\t" + "Layer Size\n"
        for _i in self._model_dict:
            if (isinstance(self._model_dict[_i], torch.nn.Linear)):
                to_print += "\t" + str(_i) + "\t\t"
                to_print += "Linear" + "\t\t" + "in_features: " + str(self._model._modules[_i].in_features) \
                 + ", out_features: " + str(self._model._modules[_i].out_features) +"\n"

            if (isinstance(self._model._modules[_i], torch.nn.ReLU)):
                to_print += "\t" + "activation" + "\t\t"
                to_print += "ReLU" + "\n"
        to_print += "\t======================================================================\n"
        return to_print

    def load_model(self, file_name):	
        self._model.load_state_dict(torch.load(file_name))

    def save_model(self, file_name = None):
        if (file_name == None):
            file_name = "trained_models/" + str(datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+".pb")
        else:
            file_name = "trained_models/" + file_name + "-" + str(datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+".pb")
        torch.save(self._model.state_dict(), file_name)
        model = self._model.cpu()
        torch.save(model.state_dict(), file_name.split(".")[0]+"_cpu.pb")

    def shuffle_data(self, test = 0):
        if (test == 0):
            n = len(self.train_data_x)
            self.train_data_x = numpy.array(self.train_data_x, dtype=numpy.float32)
            self.train_data_y = numpy.array(self.train_data_y, dtype=numpy.float32)
            self.train_data_x, self.train_data_y = shuffle(self.train_data_x, self.train_data_y, random_state=42)
        else:
            n = len(self.test_data_x)
            self.test_data_x = numpy.array(self.test_data_x)
            self.test_data_y = numpy.array(self.test_data_y)
            self.test_data_x, self.test_data_y = shuffle(self.test_data_x, self.test_data_y, random_state=42)

    def split_train_data(self, test_size = 0.1):
        self.train_data_x, self.train_data_y, self.test_data_x, self.test_data_y = train_test_split(self.train_data_x, self.train_data_y, test_size=test_size, random_state=42)

    def to_numpy(self):
        self.train_data_x = numpy.array(self.train_data_x)
        self.train_data_y = numpy.array(self.train_data_y)
        self.test_data_x = numpy.array(self.test_data_x)
        self.test_data_y = numpy.array(self.test_data_y)

    def to_torch(self):
        self.train_data_x = torch.from_numpy(self.train_data_x)
        self.train_data_y = torch.from_numpy(self.train_data_y)
        self.test_data_x = torch.from_numpy(self.test_data_x)
        self.test_data_y = torch.from_numpy(self.test_data_y)
        if (self.using_gpu == 1):
            #print("Data to cuda!")
            self.train_data_x = self.train_data_x.cuda()
            self.train_data_y  = self.train_data_y.cuda()
            self.test_data_x = self.test_data_x.cuda()
            self.test_data_y = self.test_data_y.cuda()
