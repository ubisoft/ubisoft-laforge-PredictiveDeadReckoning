from DataLoader.DataLoader import DataLoader
from NN_Model.CollisionModel import CollisionModel
from time import time
from plot.plot_path import Plotting
from util.VehicleState import VehicleState
from util.Vector3 import Vector3
from math import isclose, pi

trained_model = CollisionModel()
trained_model.init_model()

trained_model.using_gpu = True
trained_model.load_model('trained_models/collision_prediction.model')


collision_file = "data/collision-test-data/2020-02-24-17-31-59_MasterCar.txt"
collision_file_no_prediction = "data/collision-test-data/2020-02-24-17-31-59_MLReplicaCar.txt"


data = DataLoader()
data_no_prediction = DataLoader()

plt = Plotting()
data.read_file(file_name=collision_file)
data_no_prediction.read_file(file_name=collision_file_no_prediction)
master_path = data_no_prediction[collision_file_no_prediction]
plt.plot_path(path=master_path, color='y', label="DeadReckoning prediction - No delay")

master_path = data[collision_file]


plt.plot_path_orientation(path=master_path, master_path=True, label="Master Path")
data.read_collision_file(file_name=collision_file)
prev_collision_point = Vector3([0, 0, 0])
# for message in data[collision_file]:
message = data[collision_file][0]
# print(data)
state = VehicleState()
state.set_time(float(message[0]))
state.set_position(message[1:4])
state.set_rotation(message[4:8])
state.set_velocity(message[8:11])
state.set_angular_velocity(message[11:14])
state.set_actions(float(message[14]), float(message[15]))
normal = Vector3(message[16:19])
point = Vector3(message[19:22])

response = []
for i in range(1, 50):
    response.append(trained_model.predict(state, normal, point, 0.02*i, 0.02))

plt.plot_path_orientation(response, color='g', label="Collision prediction")
plt.show(xlim=[-5, 0], ylim=[130, 150])
prev_collision_point = Vector3(point)
