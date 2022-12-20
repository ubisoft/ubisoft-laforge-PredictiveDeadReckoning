import sys
from DataLoader.DataLoader import DataLoader
from NN_Model.PredictionModel import PredictionModel
from NN_Model.LSTMPredictionModel import LSTMPredictionModel
from NN_Model.BlendingModel import BlendingModel
from time import time
from plot.plot_path import Plotting
from NN_Model.PredictionAlgorithm import PredictionAlgorithm, BlendingAlgorithm
from NN_Model.DRPrediction import DRPrediction, DRBlending
from util.VehicleState import VehicleState
from math import sqrt
import numpy as np
import argparse
import os

# FILE_NAME = 'training-data/2019-09-20-13-33-35_MasterCar.txt'  # file to read from
SEND_RATE = 0.4  # frequency of the messages
DELTA_TIME = 0.02  # time between consecutive frames
PLOT_START = 0
PLOT_END = 3000

FILE_NAME = 'training-data-with-collisions/2020-04-28-13-44-25_MasterCar.txt'
SRC_DATA_DIR = 'training-data'  # 'test'

single_model = './trained_models/single_predictor.model'
lstm_model = './trained_models/lstm_predictor.model'


def main(args):

    rng = np.random.default_rng(args.seed)

    data_files = os.listdir(SRC_DATA_DIR)
    data_files = [f for f in data_files if f.endswith('MasterCar.txt')]
    rng.shuffle(data_files)

    data_files = ['2019-09-20-13-33-35_MasterCar.txt', ]
    seeds = rng.integers(0, 10000, 1)

    with open('NN-rmse-results.csv', 'w') as fp:
        fp.write('model,file,method,rmse,max-error,send-rate,drop-rate\n')

        for seed in seeds:
            for send_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                for file in data_files[:15]:
                    fce, max_fce, nne, max_nne, lstme, max_lstme = run_test(
                        datafile=file, send_rate=send_rate, drop_rate=args.drop_rate, seed=seed)

                    def format_line(method, error, max_error):
                        return '{},{},{},{},{},{}\n'.format(file, method, error, max_error, send_rate, args.drop_rate)
                    if fce is not None:
                        data = format_line('fc', fce[-1], max_fce) + format_line('nn', nne[-1], max_nne) + format_line('lstm', lstme[-1], max_lstme)
                        fp.write(data)
                        fp.flush()


def run_test(datafile, send_rate=0.4, drop_rate=0, seed=1):

    rng = np.random.default_rng(seed)

    data = DataLoader()
    full_path = os.path.join(SRC_DATA_DIR, datafile)
    data.read_file(full_path)
    master_path = data[full_path][PLOT_START:PLOT_END]

    if len(master_path) <= 0:
        return None, None, None, None, None, None

    try:
        trained_model = PredictionModel(send_rate=0.4, device='cuda')
        trained_model.load_model(single_model)
        trained_model._model.float()
    except RuntimeError as e:
        print("Unable to load {}:{}".format(single_model, e))
        return None, None, None, None, None, None

    try:
        trained_lstm_model = LSTMPredictionModel(send_rate=0.4, device='cuda')
        trained_lstm_model.init_model()
        trained_lstm_model.load_model(lstm_model)
    except RuntimeError as e:
        print("Unable to load {}: {}".format(lstm_model, e))
        return None, None, None, None, None, None

    DeadReckoning_prediction_alg = DRPrediction()
    fc_prediction = PredictionAlgorithm(DeadReckoning_prediction_alg)

    nn_prediction = PredictionAlgorithm(trained_model)
    lstm_prediction = PredictionAlgorithm(trained_lstm_model)

    plt = Plotting()
    _path_state_count = 0
    messages = [master_path[_path_state_count]]
    model_predicted_positions = []
    lstm_model_predicted_positions = []
    DeadReckoning_predicted_positions = []

    frame = 0
    master_frame = 0
    master_frame_interval = send_rate / DELTA_TIME

    last_message = master_path[master_frame]

    # store the paths generated for each of the methods
    nn_rcvd_path = [last_message]
    fc_rcvd_path = [last_message]
    lstm_rcvd_path = [last_message]

    while True:
        frame += 1  # next frame
        if frame >= len(master_path):
            # done
            break

        frame_time = frame * DELTA_TIME

        # receive a new frame
        if not frame % master_frame_interval:

            if rng.random() < drop_rate:
                # drop this message
                #
                print('#', end='', flush=True)
            else:
                print('.', end='', flush=True)
                last_message = master_path[frame]

                nn_rcvd_path.append(last_message)
                fc_rcvd_path.append(last_message)
                lstm_rcvd_path.append(last_message)

                lstm_model_predicted_positions.append(last_message)
                DeadReckoning_predicted_positions.append(last_message)
                model_predicted_positions.append(last_message)

                # don't need to predict -- we just got a message!
                continue

        predicted_state = fc_prediction.predict(message=fc_rcvd_path[-1],
                                                time_to_predict=frame_time - fc_rcvd_path[-1].get_time(),
                                                delta_time=DELTA_TIME)
        DeadReckoning_predicted_positions.append(predicted_state)

        predicted_state = nn_prediction.predict(message=fc_rcvd_path[-1],
                                                time_to_predict=frame_time - nn_rcvd_path[-1].get_time(),
                                                delta_time=DELTA_TIME)
        model_predicted_positions.append(predicted_state)

        predicted_state = lstm_prediction.predict(message=lstm_rcvd_path[-3:],
                                                  time_to_predict=frame_time - lstm_rcvd_path[-1].get_time(),
                                                  delta_time=DELTA_TIME)
        lstm_model_predicted_positions.append(predicted_state)

    # error calculations
    DeadReckoning_error = []
    DeadReckoning_rmse = []
    trained_error = []
    trained_rmse = []
    trained_lstm_error = []
    trained_lstm_rmse = []

    def squared_error(a, b):
        diff = a.get_position() - b.get_position()
        return diff.dot(diff)

    for (master, fc, nn, lstm) in zip(master_path[1:], DeadReckoning_predicted_positions, model_predicted_positions, lstm_model_predicted_positions):
        DeadReckoning_error.append(squared_error(master, fc))
        trained_error.append(squared_error(master, nn))
        trained_lstm_error.append(squared_error(master, lstm))

    fc_total_error = 0
    nn_total_error = 0
    lstm_total_error = 0
    for i in range(len(DeadReckoning_error)):
        fc_total_error += DeadReckoning_error[i]
        DeadReckoning_rmse.append(sqrt(fc_total_error / (i+1)))

        nn_total_error += trained_error[i]
        trained_rmse.append(sqrt(nn_total_error / (i+1)))
        nn_rmse_ratio = [n / d for n, d in zip(trained_rmse, DeadReckoning_rmse)]

        lstm_total_error += trained_lstm_error[i]
        trained_lstm_rmse.append(sqrt(lstm_total_error / (i+1)))
        lstm_rmse_ratio = [n / d for n, d in zip(trained_lstm_rmse, DeadReckoning_rmse)]

    outfile = "test_results/{}-rmse-error.csv".format(datafile)
    with open(outfile, 'w') as fp:
        fp.write('method,t,run,err,ratio\n')
        for i in range(1, len(DeadReckoning_error)):
            t = master_path[i+1].get_time()
            fp.write('fc'+','+str(t)+','+str(1)+','+str(DeadReckoning_rmse[i])+','+str(1)+'\n')
            fp.write('nn'+','+str(t)+','+str(1)+','+str(trained_rmse[i])+','+str(nn_rmse_ratio[i])+'\n')
            fp.write('lstm'+','+str(t)+','+str(1)+','+str(trained_lstm_rmse[i])+','+str(lstm_rmse_ratio[i])+'\n')

    print('done')
    return DeadReckoning_rmse, sqrt(max(DeadReckoning_error)), trained_rmse, sqrt(max(trained_error)), trained_lstm_rmse, sqrt(max(trained_lstm_error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Data Converter -- translate dataset of positions into triplets for learning")
    parser.add_argument(
        '-f', '--file', default=FILE_NAME,
        help='Test data to load')

    parser.add_argument(
        '-s', '--send-rate', default=0.4, type=float,
        help='time separation between messages')

    parser.add_argument(
        '--seed', default=42, type=int,
        help='Seed for random generator (for repeatability)')

    parser.add_argument(
        '-d', '--drop-rate', default=0.0, type=float,
        help='Probability of message drop')

    args = parser.parse_args()

    main(args)
#
