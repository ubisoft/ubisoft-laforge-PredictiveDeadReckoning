from NN_Model.Collision_NN_Model import NeuralNetworkModel
from util.VehicleState import VehicleState
from util.Vector3 import Vector3
from util.Quaternion import Quaternion
from tqdm import tqdm
from time import time
import torch
import numpy


class CollisionModel(NeuralNetworkModel):
    def __init__(self, send_rate=0.2, data=None, num_inputs=10):
        NeuralNetworkModel.__init__(self, num_inputs=num_inputs, num_outputs=7)
        self._send_rate = send_rate
        self._data = data

    def train(self, num_epochs=100, n_batch_size=50000, data=None):
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

            for batch_no in range(num_batches - 1):
                batch_x = self.train_data_x[batch_no*n_batch_size:(batch_no+1)*n_batch_size]
                batch_y = self.train_data_y[batch_no*n_batch_size:(batch_no+1)*n_batch_size]
                output = self._model(batch_x)
                loss = criterion(output, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("\n\t epoch/num_epochs: [%d/%d] loss:%f" % (epoch, num_epochs, loss.item()))

        print("\tTraining model finished in %f seconds" % (time() - s_time))

    def predict(self, message, normal, point, time_to_predict, delta_time):
        curr_pos = message.get_position()
        curr_rot = message.get_rotation()
        curr_vel = message.get_velocity()
        curr_ang_vel = message.get_angular_velocity()
        curr_action_h = message.get_action_h()
        curr_action_v = message.get_action_v()
        master_vel = curr_rot.rotate(curr_vel)
        # vel_mag = master_vel.magnitude()
        # master_vel = master_vel.normalized()
        master_ang_vel = curr_rot.rotate(curr_ang_vel)
        point = curr_rot.rotate(point - curr_pos)
        normal = curr_rot.rotate(normal)
        relative_normal = Vector3(normal).angle_about_axis(curr_vel, "y")
        x = [
            master_vel[0],
            master_vel[1],
            master_vel[2],
            # vel_mag,
            normal[0],
            normal[1],
            normal[2],
            point[0],
            point[1],
            point[2],
            time_to_predict
        ]
        result = self._model(torch.tensor(x)).tolist()
        displacement = Vector3(vec=result[0:3])
        displacement = curr_rot.conjugate().rotate(displacement)
        orientation = curr_rot*Quaternion(vec=result[3:7])
        orientation = orientation.normalized()
        predicted_pos = message.get_position() + displacement
        predicted_state = VehicleState()
        predicted_state.set_time(message.get_time() + time_to_predict)
        predicted_state.set_position(predicted_pos)
        predicted_state.set_rotation(orientation)
        predicted_state.set_velocity(message.get_velocity())
        predicted_state.set_angular_velocity(message.get_angular_velocity())
        return predicted_state

    def predict_path(self, message, normal, point, time_to_predict, delta_time):
        predicted_path = []
        predicted_path.append(message)
        intervals = int(time_to_predict/delta_time + 0.5)
        for i in range(1, intervals):
            predicted_path.append(self.predict(message, normal, point, delta_time*i, delta_time))
        return predicted_path

    def format_data(self, data=None, test=0):
        '''
        constructing the training data
        '''
        look_ahead = 0.5
        new_collision_flg = False
        for _file_name in data._data:
            print(_file_name)
            for _i in tqdm(range(len(data._data[_file_name])-1)):

                curr_time = data._data[_file_name][_i][0]
                collision_data = []
                if abs(curr_time - 0.0) < 0.0001:
                    collision_data = data._data[_file_name][_i]
                    collision_pos = Vector3(collision_data[1:4])
                    curr_rot = Quaternion(collision_data[4:8])
                    curr_vel = Vector3(collision_data[8:11])
                    curr_action_h = collision_data[14]
                    curr_action_v = collision_data[15]
                    master_coord_vel = curr_rot.rotate(curr_vel)
                    # vel_mag = master_coord_vel.magnitude()
                    # master_coord_vel = master_coord_vel.normalized()
                    contact_normal = collision_data[16:19]
                    contact_normal = curr_rot.rotate(contact_normal)
                    contact_point = Vector3(collision_data[19:22])
                    contact_point = curr_rot.rotate(contact_point - collision_pos)
                    relative_normal = Vector3(contact_normal).angle_about_axis(curr_vel, "y")
                    new_collision_flg = True

                if new_collision_flg:
                    for _j in range(_i, len(data._data[_file_name])):
                        if (_j >= len(data._data[_file_name])):
                            break
                        if data._data[_file_name][_j][0] > look_ahead:
                            break

                        next_pos = Vector3(data._data[_file_name][_j][1:4])
                        next_orientation = Quaternion(data._data[_file_name][_j][4:8])
                        delta_rot = curr_rot.inverse()*next_orientation
                        displacement = curr_rot.rotate(next_pos - collision_pos)

                        x_train = [
                            master_coord_vel[0],
                            master_coord_vel[1],
                            master_coord_vel[2],
                            contact_normal[0],
                            contact_normal[1],
                            contact_normal[2],
                            contact_point[0],
                            contact_point[1],
                            contact_point[2],
                            data._data[_file_name][_j][0]
                        ]
                        y_train = [Vector3(data._data[_file_name][_j][1:4])]
                        next_orientation = Quaternion(data._data[_file_name][_j][4:8])
                        delta_rot = curr_rot.inverse()*next_orientation
                        displacement = curr_rot.rotate(next_pos - collision_pos)

                        x_train = [
                            master_coord_vel[0],
                            master_coord_vel[1],
                            master_coord_vel[2],
                            contact_normal[0],
                            contact_normal[1],
                            contact_normal[2],
                            contact_point[0],
                            contact_point[1],
                            contact_point[2],
                            data._data[_file_name][_j][0]
                        ]
                        y_train = [
                            displacement[0],
                            displacement[1],
                            displacement[2],
                            delta_rot[0],
                            delta_rot[1],
                            delta_rot[2],
                            delta_rot[3],
                        ]

                        if (test == 0):
                            self.train_data_x.append(x_train)
                            self.train_data_y.append(y_train)
                        else:
                            self.test_data_x.append(x_train)
                            self.test_data_y.append(y_train)

                        if (test == 0):
                            self.train_data_x.append(x_train)
                            self.train_data_y.append(y_train)
                        else:
                            self.test_data_x.append(x_train)
                            self.test_data_y.append(y_train)
                    new_collision_flg = False
                    _i = _j
