from NN_Model.Blending_NN_Model import NeuralNetworkModel
from util.VehicleState import VehicleState
from util.Vector3 import Vector3
from tqdm import tqdm
from time import time
import torch
import numpy


class BlendingModel(NeuralNetworkModel):
    def __init__(self, send_rate=0.2, data=None):
        NeuralNetworkModel.__init__(self, num_inputs=10, num_outputs=3)
        self._send_rate = send_rate
        self._data = data

    def train(self, num_epochs=300, n_batch_size=1000, data=None):
        '''
        training model
        num_epochs: number of epochs for Training
        n_batch_size: batch size at each iteration
        '''
        if (self._model == 0):
            self.init_model()
        if data == None and self._data == None:
            print("No data to train!")
            return

        if data == None:
            data = self._data

        print(self.__str__())

        print("Construcing training data!")
        s_time = time()
        self.format_data(data=data)
        print("\tConstrucing training data finished in %f seconds" % (time() - s_time))

        print("Shuffling training data!")
        s_time = time()
        self.shuffle_data()
        print("\tShuffling training data finished in %f seconds" % (time() - s_time))

        self.to_numpy()
        self.to_torch()

        if (self.using_gpu and torch.cuda.is_available()):
            self._model.cuda()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        print("Size of training data: %d" % len(self.train_data_x))
        print("Training model!")
        s_time = time()
        num_batches = int(len(self.train_data_x)/n_batch_size)
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            # output = self._model(self.train_data_x)
            # loss = criterion(output, self.train_data_y)
            # print(loss.item())
            for batch_no in range(num_batches - 1):
                batch_x = self.train_data_x[batch_no*n_batch_size:(batch_no+1)*n_batch_size]
                batch_y = self.train_data_y[batch_no*n_batch_size:(batch_no+1)*n_batch_size]
                output = self._model(batch_x)
                loss = criterion(output, batch_y)
                if (batch_no % 5 == 0 and batch_no > 0):
                    print("\n\t epoch/num_epochs: [%d/%d] batch_no/num_batches: [%d/%d] loss:%f" % (epoch, num_epochs, batch_no, num_batches, loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print("\tTraining model finished in %f seconds" % (time() - s_time))

    def blend(self, current_state, predicted_state, blend_time, delta_time):
        current_pos = current_state.get_position()
        current_vel = current_state.get_velocity()
        predicted_pos = predicted_state.get_position()
        predicted_vel = predicted_state.get_velocity()
        dx = predicted_pos - current_pos
        dv = predicted_vel - current_vel
        dx_norm = dx.normalized()
        dx_mag = dx.magnitude()
        dv_norm = dv.normalized()
        dv_mag = dv.magnitude()
#        x = [
#                current_vel[0],
#                current_vel[1],
#                current_vel[2],
#                dx_norm[0],
#                dx_norm[1],
#                dx_norm[2],
#                (dx_mag - 3.3102)/6.88008,
#                dv_norm[0],
#                dv_norm[1],
#                dv_norm[2],
#                (dv_mag - 6.979431)/33.9688,
#                blend_time
#        ]
        x = [
            current_vel[0],
            current_vel[1],
            current_vel[2],
            dx[0],
            dx[1],
            dx[2],
            dv[0],
            dv[1],
            dv[2],
            blend_time
        ]
        output = self._model(torch.tensor(x)).tolist()
#        normalized = Vector3([output[0], output[1], output[2]]).normalized()
#        magnitude = output[3]*0.005445+0.38391
#
#        v = normalized*magnitude
        v = Vector3(output)
        smooth_state = VehicleState()
        smooth_state.set_time(current_state.get_time() + delta_time)
        smooth_state.set_position(current_pos + (current_vel + v)*delta_time)
        smooth_state.set_rotation(current_state.get_rotation())
        smooth_state.set_velocity(current_vel + v)
        smooth_state.set_angular_velocity(current_state.get_angular_velocity())
        return smooth_state

    def format_data(self, data=None, test=0):
        '''
        constructing the training data
        '''
        for _file_name in data._data:
            for _i in tqdm(range(len(data._data[_file_name])-1)):
                current_vel = Vector3(vec=data._data[_file_name][_i][:3])
                dx = Vector3(vec=data._data[_file_name][_i][3:6])
                dv = Vector3(vec=data._data[_file_name][_i][6:9])
                blend_time = data._data[_file_name][_i][9]
                dx_norm = dx.normalized()
                dx_mag = dx.magnitude()
                dv_norm = dv.normalized()
                dv_mag = dv.magnitude()
#                x_train = [
#                        current_vel[0],
#                        current_vel[1],
#                        current_vel[2],
#                        dx_norm[0],
#                        dx_norm[1],
#                        dx_norm[2],
#                        dx_mag,
#                        dv_norm[0],
#                        dv_norm[1],
#                        dv_norm[2],
#                        dv_mag,
#                        blend_time
#                ]
                x_train = [
                    current_vel[0],
                    current_vel[1],
                    current_vel[2],
                    dx[0],
                    dx[1],
                    dx[2],
                    dv[0],
                    dv[1],
                    dv[2],
                    blend_time
                ]

                dy = Vector3([data._data[_file_name][_i][10],
                              data._data[_file_name][_i][11],
                              data._data[_file_name][_i][12]])

                dy_mag = dy.magnitude()
                dy_norm = dy.normalized()

#                y_train = [
#                        dy_norm[0],
#                        dy_norm[1],
#                        dy_norm[2],
#                        dy_mag
#                ]
                y_train = [
                    data._data[_file_name][_i][10] - current_vel[0],
                    data._data[_file_name][_i][11] - current_vel[1],
                    data._data[_file_name][_i][12] - current_vel[2]
                ]
                if (test == 0):
                    self.train_data_x.append(x_train)
                    self.train_data_y.append(y_train)
                else:
                    self.test_data_x.append(x_train)
                    self.test_data_y.append(y_train)

        self.train_data_x = numpy.array(self.train_data_x)
        self.train_data_y = numpy.array(self.train_data_y)

#        x = numpy.array([self.train_data_x[_i][6] for _i in range(len(self.train_data_x))])
#        mean_x = numpy.mean(x)
#        sigma_x = numpy.var(x)
#        for _i in range(len(self.train_data_x)):
#            self.train_data_x[_i][6] = (self.train_data_x[_i][6] - mean_x)/sigma_x
#
#        print(mean_x, sigma_x)
#        x = numpy.array([self.train_data_x[_i][10] for _i in range(len(self.train_data_x))])
#        mean_x = numpy.mean(x)
#        sigma_x = numpy.var(x)
#        for _i in range(len(self.train_data_x)):
#            self.train_data_x[_i][10] = (self.train_data_x[_i][10] - mean_x)/sigma_x
#        print(mean_x, sigma_x)
#        y = numpy.array([self.train_data_y[_i][3] for _i in range(len(self.train_data_y))])
#        mean_y = numpy.mean(y)
#        sigma_y = numpy.var(y)
#        for _i in range(len(self.train_data_y)):
#            self.train_data_y[_i][3] = (self.train_data_y[_i][3] - mean_y)/sigma_y
#        print(mean_y, sigma_y)
