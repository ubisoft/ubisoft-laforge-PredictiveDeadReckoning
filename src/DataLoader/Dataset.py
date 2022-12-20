import torch
import os
import pickle
#https://discuss.pytorch.org/t/loading-huge-data-functionality/346/2
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data_files = os.listdir('data_dir')

    def __getindex__(self, idx):
        return self.load_file(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)

    def load_file(filename):
        file = open(filename,'rb')
        (X, Y) = pickle.load(file)
        file.close()
        return (X, Y)