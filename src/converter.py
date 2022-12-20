#
# A Python program to convert the raw trajectory data and process it into a form suitable for training.
#
# Note that the source and destination directories a currently fixed (see below)
#

import argparse
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
import uuid

# internal modules
from util.VehicleState import VehicleState


# source directory for training data -- these directories must exist
inputs = [
    './data/split-training-data/split-train-data',
    './data/split-training-data/split-test-data'
]

# destination directory  -- these directories must exist
outputs = [
    'single-split-train-data',
    'single-split-test-data'
]


def parse_vec(vec):
    vector = []
    vec = vec.split("(")[1]
    vec = vec.split(")")[0]
    vec = vec.split(",")
    for _i in range(len(vec)):
        vector.append(round(float(vec[_i]), 5))
    return vector


def read_file(filename):
    assert (filename is not None)

    vehicle_data = []

    try:
        with open(filename, 'r') as fp:

            for line in fp:
                data = line.split("|")
                if len(data) < 7:
                    # skip everything but the actual data lines
                    continue

                state = VehicleState()
                state.set_time(float(data[0]))
                state.set_position(parse_vec(data[1]))
                state.set_rotation(parse_vec(data[2]))
                state.set_velocity(parse_vec(data[3]))
                state.set_angular_velocity(parse_vec(data[4]))
                state.set_actions(float(data[5]), float(data[6]))
                vehicle_data.append(state)

    except IOError:
        print("ERROR: Failed to read data file: {}".format(filename))

    return vehicle_data


def write_data(output_dir, send_rate, x_data, y_data):
    out_name = os.path.join(output_dir, 'sr'+str(send_rate)+'-'+str(uuid.uuid4()) + '.pkl')
    torched_x = torch.cat(x_data, dim=0)
    torched_y = torch.cat(y_data, dim=0)
    torched_y = torch.unsqueeze(torched_y, dim=1)

    with open(out_name, 'wb') as fp:
        pickle.dump((torched_x, torched_y), fp)


def convert(in_file, output_dir, samples_per_file=100000, depth=3, send_rate=0.4):
    '''
    Convert a position based data file into a series of sets, with a set
    generating up to send_rate / frame_rate entries

    :params:
    in_file: raw data file
    out_file: pickled tensor of vehicle triplets
    depth: number of messages required per set -- default to 3
    send_rate: the time interval between messages
    '''

    try:
        vehicle_data = read_file(in_file)
        if not len(vehicle_data):
            print("Datafile {} empty -- skipping".format(in_file))
            return
    except ValueError:
        print("Datafile {} unreadable -- skipping".format(in_file))
        return

    x_training_data = []
    y_training_data = []

    # frame_rate is assumed to be consistent for the file and is the distance between
    # any two messages.  Choose the first two for convenience
    frame_rate = vehicle_data[1].get_time() - vehicle_data[0].get_time()
    frames_per_interval = int(send_rate / frame_rate)

    for current_frame in tqdm(range(len(vehicle_data) - (depth * frames_per_interval))):

        end_frame = current_frame + (depth - 1) * frames_per_interval + 1
        frames = [f for f in vehicle_data[current_frame: end_frame: frames_per_interval]]

        x_train = []

        # use the last frame as the position reference -- everything should be relative to it
        curr_time = frames[-1].get_time()
        curr_pos = frames[-1].get_position()
        curr_rot = frames[-1].get_rotation()

        for frame in frames:
            frame_data = []
            frame_data.extend(curr_rot.conjugate().rotate(frame.get_velocity()))

            # TODO: ??? Rotating the angular_velocity -- why? the rate of rotation doesn't vary with the frame...
            frame_data.extend(frame.get_angular_velocity())
            frame_data.append(frame.get_action_h())
            frame_data.append(frame.get_action_v())
            frame_data.append(curr_time - frame.get_time())

            x_train.append(frame_data)

        x_train = np.array(x_train)

        for future_step in range(1, frames_per_interval + 1):
            future_x = x_train.copy()
            future_x[:, -1] += future_step * frame_rate

            try:
                future_frame = vehicle_data[current_frame + (depth-1) * frames_per_interval + future_step]
            except IndexError:
                # at the end of the data
                break

            if args.full_state:
                future_state = []
                future_state.extend(curr_rot.conjugate().rotate(future_frame.get_velocity()))
                future_state.extend(frame.get_angular_velocity())
                future_state.append(future_frame.get_action_h())
                future_state.append(future_frame.get_action_v())
                future_state.append(curr_time - future_frame.get_time())
                future_y = np.array(future_state)
            else:
                future_y = np.array(curr_rot.conjugate().rotate(future_frame.get_position() - curr_pos))

            x_training_data.append(torch.unsqueeze(torch.tensor(future_x), dim=0))
            y_training_data.append(torch.unsqueeze(torch.tensor(future_y), dim=0))

        if len(x_training_data) > samples_per_file:
            write_data(output_dir=output_dir, send_rate=send_rate, x_data=x_training_data, y_data=y_training_data)
            x_training_data = []
            y_training_data = []

    if len(x_training_data):
        write_data(output_dir=output_dir, send_rate=send_rate, x_data=x_training_data, y_data=y_training_data)


def convert_files(args):
    for in_dir, out_dir in zip(inputs, outputs):
        data_files = [f for f in os.listdir(in_dir) if f.endswith('MasterCar.txt')]

        for file in tqdm(data_files, desc='Converting...'):
            in_file = os.path.join(in_dir, file)

            convert(in_file, out_dir, samples_per_file=100000, depth=args.depth, send_rate=args.send_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vehicle Data Converter -- translate dataset of positions into triplets for learning")
    parser.add_argument(
        '-s', '--send-rate', default=0.4, type=float,
        help='time separation between messages')
    parser.add_argument(
        '--full-state',
        action='store_true',
        help='full state prediction information')

    parser.add_argument(
        '-d', '--depth', default=3, type=int,
        help='Number of messages to use for prediction')

    args = parser.parse_args()

    convert_files(args)
