from DataLoader.DataLoader import DataLoader
from NN_Model.CollisionModel import CollisionModel
from time import time
from plot.plot_path import Plotting
from util.VehicleState import VehicleState
from util.Vector3 import Vector3

import argparse


def main(args):
    data = DataLoader()
    s_time = time()
    data.read_all_files_in_folder(folder_name="collision-training-data/")
    print("\tReading files finished in %f seconds" % (time() - s_time))

    model = CollisionModel(num_inputs=10)
    model.using_gpu = args.gpu
    model.train(data=data)
    model.save_model()

    collision_file = "collision-test-data/2020-02-25-10-02-43_MasterCar.txt"
    data.read_file(file_name=collision_file)

    plt = Plotting()
    master_path = data[collision_file]
    plt.plot_path(path=master_path)
    data.read_collision_file(file_name=collision_file)
    message = data[collision_file][0]

    state = VehicleState()
    state.set_time(float(message[0]))
    state.set_position(message[1:4])
    state.set_rotation(message[4:8])
    state.set_velocity(message[8:11])
    state.set_angular_velocity(message[11:14])
    state.set_actions(float(message[14]), float(message[15]))
    normal = state._rotation.rotate(Vector3(message[16:19]))
    point = Vector3(message[19:22])
    response = model.predict_path(state, normal, point, 0.5, 0.02)

    plt.plot_path(response, color='g')
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--gpu',
        action='store_true',
        help='Use the GPU (nVidia only)')

    args = argparser.parse_args()

    main(args)
