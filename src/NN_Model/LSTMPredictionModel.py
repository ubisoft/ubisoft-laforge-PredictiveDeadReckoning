from NN_Model.LSTM_Model import LSTM_Model
from util.VehicleState import VehicleState
from util.Vector3 import Vector3
from tqdm import tqdm
from time import time
from os import listdir
import os
import shutil
import math
import torch
import torch.utils.data
import uuid
import pickle
import numpy
import datetime
import numpy as np
from os.path import isfile
from util.Quaternion import Quaternion
from util.Vector3 import Vector3

model_parameters_file = './lstm_predictor.model'
data_dir = './lstm-split-train-data'


class LSTMPredictionModel:
    def __init__(self, send_rate=0.2, data=None, device='cuda'):

        self.input_dim = 9
        self.output_dim = 3

        self._model = LSTM_Model(input_dim=self.input_dim, output_dim=self.output_dim, hidden_dim=128, depth=2, device=device)
        self._send_rate = send_rate
        self._data = data
        self._device = device

    def init_model(self):
        pass

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

    def train(self, num_epochs=300, n_batch_size=512, roll_count=10, test_size=10, seed=42, data=None):
        '''
        training model
        num_epochs: number of epochs for Training
        n_batch_size: batch size at each iteration
        '''
        print(self.__str__())

        file_batch_size = n_batch_size

        dataset = LSTMDataset(data_dir, test_size=test_size, seed=seed)
        test_len = len(dataset) // 10                    # 10% test data
        train_len = len(dataset) - test_len              # and the remainder for training

        datasets = torch.utils.data.random_split(dataset, (train_len, test_len), generator=torch.Generator().manual_seed(0))
        train_dataloader = torch.utils.data.DataLoader(datasets[0], num_workers=0, batch_size=file_batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(datasets[1], num_workers=0, batch_size=file_batch_size, shuffle=True)

        learning_rate = 1e-4

        if (self.using_gpu and torch.cuda.is_available()):
            self._model.cuda()

        self._model.float()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        if isfile(model_parameters_file):
            self._model.load_state_dict(torch.load(model_parameters_file))
            self.min_test_loss = self.test_model(dataloader=test_dataloader, criterion=criterion)
        else:
            self.min_test_loss = np.inf

        print("Training model!")
        s_time = time()

        for epoch in tqdm(range(num_epochs), desc='Epochs'):

            for rn in range(roll_count):

                if rn:
                    # move to the next dataset group
                    dataset.roll()
                    test_len = len(dataset) // 10                    # 10% test data
                    train_len = len(dataset) - test_len              # and the remainder for training

                    datasets = torch.utils.data.random_split(dataset, (train_len, test_len), generator=torch.Generator().manual_seed(rn))
                    train_dataloader = torch.utils.data.DataLoader(datasets[0], num_workers=0, batch_size=file_batch_size, shuffle=True)
                    test_dataloader = torch.utils.data.DataLoader(datasets[1], num_workers=0, batch_size=file_batch_size, shuffle=True)

                # initialize for new test
                self.avg_test_loss = 0
                self.avg_train_loss = 0

                for (x, y) in tqdm(train_dataloader, desc='Training'):

                    # reshape X and Y
                    x = x.reshape(-1, 3, self.input_dim)
                    y = y.reshape(-1, 1, self.output_dim)

                    # initialize the gradient
                    optimizer.zero_grad()
                    y_pred = self._model(x)
                    loss = criterion(y_pred.squeeze(), y.squeeze())
                    loss.backward()
                    optimizer.step()

                    self.avg_train_loss += loss

                self.avg_train_loss /= len(train_dataloader)
                self.avg_test_loss = self.test_model(dataloader=test_dataloader, criterion=criterion)

                # lr_scheduler.step(self.avg_test_loss)
                lr = optimizer.param_groups[0]["lr"]

                if self.avg_test_loss < self.min_test_loss:
                    torch.save(self._model.state_dict(), model_parameters_file)
                    self.min_test_loss = self.avg_test_loss

        print("\tTraining model finished in %f seconds" % (time() - s_time))

    def test_model(self, dataloader, criterion):
        loss = 0
        for (x, y) in tqdm(dataloader, desc='Testing'):
            with torch.no_grad():
                # reshape X and Y
                # TODO: hard coded depth/history for the time being
                x = x.reshape(-1, 3, self.input_dim)
                y = y.reshape(-1, 1, self.output_dim)

                y_pred = self._model(x)
                # just compare the last position/output
                loss = criterion(y_pred.squeeze(), y.squeeze())
                loss += loss

        loss /= len(dataloader)
        return loss

    def predict(self, message, time_to_predict, delta_time, full_state=False):
        x_test = []
        curr_time = message[-1].get_time()
        curr_pos = message[-1].get_position()
        curr_rot = message[-1].get_rotation()
        curr_vel = message[-1].get_velocity()
        curr_ang_vel = message[-1].get_angular_velocity()

        for m in message:
            frame_data = []
            frame_data.extend(curr_rot.conjugate().rotate(m.get_velocity()))
            frame_data.extend(m.get_angular_velocity())
            frame_data.append(m.get_action_h())
            frame_data.append(m.get_action_v())
            frame_data.append(curr_time - m.get_time() + time_to_predict)

            x_test.append(frame_data)

        x_test = np.array(x_test).astype(np.float32)

        # TODO: if there isn't enough data, repeat the first message until there is
        while x_test.shape[0] < 3:
            # print("WARNING: Repeating first message to fill buffer")
            x_test = np.vstack([x_test[0, :], x_test])

        if not full_state:
            displacement = Vector3(vec=torch.squeeze(self._model(torch.unsqueeze(torch.tensor(x_test), dim=0).to(self._device))).tolist())
            displacement = curr_rot.rotate(displacement)
            predicted_state = VehicleState()
            predicted_state.set_time(message[-1].get_time() + time_to_predict)
            predicted_state.set_position(curr_pos + displacement)
            predicted_state.set_rotation(curr_rot)
            predicted_state.set_velocity(curr_vel)
            predicted_state.set_angular_velocity(curr_ang_vel)
        else:
            prediction = torch.squeeze(self._model(torch.unsqueeze(torch.tensor(x_test), dim=0).to(self._device))).tolist()
            predicted_state = VehicleState()
            predicted_state.set_velocity(curr_rot.rotate(Vector3(vec=prediction[0:3])))
            predicted_state.set_angular_velocity(Vector3(vec=prediction[3:6]))
            predicted_state.set_actions(action_h=prediction[6], action_v=prediction[7])
            predicted_state.set_time(prediction[8])

            # TODO: If we ever come back to this -- the current rotation should be altered by
            #       the angular velocity x time, scaling the cur_ang_vel vector -- that's not
            #       done on the next two lines.  Instead of debugging, we'll leave it for future
            #       work...
            # vec = time_to_predict * predicted_state.get_angular_velocity()
            # w = Quaternion([vec[0], vec[1], vec[2], 0]).normalized()
            # predicted_state.set_rotation(w * curr_rot * w.inverse())
            predicted_state.set_rotation(curr_rot)

            displacement = curr_pos + predicted_state.get_velocity() * time_to_predict
            predicted_state.set_position(displacement)

        return predicted_state

    def predict_path(self, messages, time_to_predict, delta_time):
        predicted_path = []
        predicted_path.append(messages[-1].get_position())
        intervals = int(time_to_predict/delta_time + 0.5)
        for i in range(1, intervals):
            predicted_path.append(self.predict(messages, delta_time*i, delta_time))
        return predicted_path


class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, test_size=10, seed=42, device='cuda'):
        self.data_dir = data_dir
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.test_size = test_size

        self.data_files = os.listdir(data_dir)
        self.data_files = [f for f in self.data_files if not (f.startswith('sr0.5') or f.startswith('sr0.6'))]

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
