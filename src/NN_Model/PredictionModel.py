from NN_Model.NN_Model import NeuralNetworkModel
from util.VehicleState import VehicleState
from util.Vector3 import Vector3
from tqdm import tqdm
from time import time
from os.path import isfile
import os
import torch
import torch.utils.data
import pickle
import numpy as np
import datetime


model_parameters_file = './single_predictor.model'
data_dir = './single-split-train-data/'


class PredictionModel:
    def __init__(self, send_rate=0.2, data=None, device='cuda'):

        self.input_dim = 9
        self.output_dim = 3

        self._model = NeuralNetworkModel(num_inputs=self.input_dim, num_outputs=self.output_dim)
        self._send_rate = send_rate
        self._data = data
        self._device = device

    def init_model(self):
        pass

    def train(self, num_epochs=300, n_batch_size=1000, roll_count=1, test_size=1, seed=42, data=None):
        '''
        training model
        num_epochs: number of epochs for Training
        n_batch_size: batch size at each iteration
        '''

        print(self._model)

        dataset = SingleDataset(data_dir=data_dir, test_size=test_size, seed=seed)
        test_len = len(dataset) // 10                    # 10% test data
        train_len = len(dataset) - test_len              # and the remainder for training

        file_batch_size = n_batch_size

        datasets = torch.utils.data.random_split(dataset, (train_len, test_len), generator=torch.Generator().manual_seed(5))
        train_dataloader = torch.utils.data.DataLoader(datasets[0], num_workers=0, batch_size=file_batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(datasets[1], num_workers=0, batch_size=file_batch_size, shuffle=True)

        learning_rate = 1e-3

        if (self._device == 'cuda' and torch.cuda.is_available()):
            self._model.cuda()

        self._model.float()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        if isfile(model_parameters_file):
            self._model.load_state_dict(torch.load(model_parameters_file))
            self.min_test_loss = self.test_model(test_dataloader, criterion=criterion)
        else:
            self.min_test_loss = np.inf

        print("Training model!")
        self._model.train()
        s_time = time()

        for epoch in tqdm(range(num_epochs), desc='Epochs'):

            if epoch:
                # move to the next dataset group
                dataset.roll()
                test_len = len(dataset) // 10                    # 10% test data
                train_len = len(dataset) - test_len              # and the remainder for training

                datasets = torch.utils.data.random_split(dataset, (train_len, test_len), generator=torch.Generator().manual_seed(5))
                train_dataloader = torch.utils.data.DataLoader(datasets[0], num_workers=0, batch_size=file_batch_size, shuffle=True)
                test_dataloader = torch.utils.data.DataLoader(datasets[1], num_workers=0, batch_size=file_batch_size, shuffle=True)

            for _ in range(roll_count):
                # initialize for new test
                self.avg_test_loss = 0
                self.avg_train_loss = 0

                for (batch_x, batch_y) in tqdm(train_dataloader, desc='Batches'):
                    # for (batch_x, batch_y) in tqdm(dataloader, desc='Batches'):

                    output = self._model(batch_x.squeeze())
                    loss = criterion(output.squeeze(), batch_y.squeeze())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    self.avg_train_loss += loss

                self.avg_train_loss /= len(train_dataloader)
                self.avg_test_loss = self.test_model(dataloader=test_dataloader, criterion=criterion)

                lr_scheduler.step(self.avg_test_loss)
                lr = optimizer.param_groups[0]["lr"]

                if self.avg_test_loss < self.min_test_loss:
                    torch.save(self._model.state_dict(), model_parameters_file)
                    self.min_test_loss = self.avg_test_loss


        print("\tTraining model finished in %f seconds" % (time() - s_time))

    def __str__(self) -> str:
        return self._model.__str__()

    def load_model(self, file_name):
        self._model.load_state_dict(torch.load(file_name))

    def save_model(self, file_name=None):
        if (file_name == None):
            file_name = "trained_models/" + str(datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+".pb")
        else:
            file_name = "trained_models/" + file_name + "-" + str(datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+".pb")
        torch.save(self._model.state_dict(), file_name)

    def test_model(self, dataloader, criterion):
        loss = 0
        self._model.eval()
        for (x, y) in tqdm(dataloader, desc='Testing'):
            with torch.no_grad():
                y_pred = self._model(x.squeeze())
                # just compare the last position/output
                loss = criterion(y_pred.squeeze(), y.squeeze())
                loss += loss

        self._model.train()
        loss /= len(dataloader)
        return loss

    def predict(self, message, time_to_predict, delta_time):

        try:
            self._model.eval()

            curr_rot = message.get_rotation()
            curr_vel = message.get_velocity()
            curr_ang_vel = message.get_angular_velocity()
            curr_action_h = message.get_action_h()
            curr_action_v = message.get_action_v()
            master_vel = curr_rot.conjugate().rotate(curr_vel)
            master_ang_vel = curr_ang_vel
            x = [
                master_vel[0],
                master_vel[1],
                master_vel[2],
                master_ang_vel[0],
                master_ang_vel[1],
                master_ang_vel[2],
                curr_action_h,
                curr_action_v,
                time_to_predict
            ]
            displacement = Vector3(vec=self._model(torch.tensor(x).reshape(1, -1).float()).squeeze().tolist())
            displacement = curr_rot.rotate(displacement)  # was rotated by conjugate()
            predicted_pos = message.get_position() + displacement
            predicted_state = VehicleState()
            predicted_state.set_time(message.get_time() + time_to_predict)
            predicted_state.set_position(predicted_pos)
            predicted_state.set_rotation(curr_rot)
            predicted_state.set_velocity(curr_vel)
            predicted_state.set_angular_velocity(curr_ang_vel)
        finally:
            self._model.train()

        return predicted_state

    def predict_path(self, message, time_to_predict, delta_time):
        predicted_path = []
        predicted_path.append(message.get_position())
        intervals = int(time_to_predict/delta_time + 0.5)
        for i in range(1, intervals):
            predicted_path.append(self.predict(message, delta_time*i, delta_time))
        return predicted_path


class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, test_size=10, seed=42, prefix=None, device='cuda'):
        self.data_dir = data_dir
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.test_size = test_size

        self.data_files = os.listdir(data_dir)

        self.data_files = [f for f in self.data_files]
        if prefix is not None:
            self.data_files = [f for f in self.data_files if f.startswith(prefix)]
        self.rng.shuffle(self.data_files)

        self.cache_dataset()

    def cache_dataset(self):
        self.X = None
        self.Y = None

        for i in range(min(len(self.data_files), self.test_size)):
            X, Y = self.load_file(self.data_files[i])
            try:
                self.X = torch.cat([self.X, X], dim=0)
                self.Y = torch.cat([self.Y, Y], dim=0)
            except TypeError:
                self.X = X
                self.Y = Y

    def roll(self):
        self.data_files = self.data_files[self.test_size:] + self.data_files[:self.test_size]
        self.cache_dataset()

    def __getitem__(self, idx):
        # cache a subset of the training files

        # BUGBUG: hack to ensure we don't have a smaller cache than expected
        idx = idx % self.X.shape[0]

        return (self.X[idx, ...], self.Y[idx, ...])

    def __len__(self):
        return self.X.shape[0]

    def load_file(self, filename):
        filename = os.path.join(self.data_dir, filename)
        with open(filename, 'rb') as file:
            (X, Y) = pickle.load(file)

        X = X.float().to(self.device)
        Y = Y.float().to(self.device)

        return (X, Y)
