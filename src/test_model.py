import sys
from DataLoader.DataLoader import DataLoader
from NN_Model.PredictionModel import PredictionModel
from NN_Model.LSTMPredictionModel import LSTMPredictionModel
from time import time
from plot.plot_path import Plotting
from NN_Model.PredictionAlgorithm import PredictionAlgorithm
from NN_Model.DRPrediction import DRPrediction
from math import sqrt
import numpy as np
import argparse
import os


SEND_RATE = 0.4  # frequency of the messages
DELTA_TIME = 0.02  # time between consecutive frames

############
# For example plot
FILE_NAME = '2019-09-20-13-33-35_MasterCar.txt'  # file to read from

##############
# Otherwise...
PLOT_START = 0
PLOT_END = 2000  # 60 second segments
SRC_DATA_DIR = 'single-split-test-data'


def main(args):

    rng = np.random.default_rng(args.seed)

    data_files = os.listdir(SRC_DATA_DIR)
    data_files = [f for f in data_files if f.endswith('MasterCar.txt')]
    rng.shuffle(data_files)

    # data_files = [FILE_NAME]

    seeds = rng.integers(0, 10000, 1)

    tests_required = 25
    tests = []
    for file in data_files:
        attempts = 0
        while attempts < 10:
            start = PLOT_START  # rng.integers(0, 500)
            path = load_file(file, start)
            if path is not None:
                tests.append((file, start, path))
                print("Loaded: {}".format(len(tests)))
                break
            attempts += 1

        if len(tests) >= tests_required:
            break

    with open('rmse-results.csv', 'w') as fp:
        fp.write('file,method,rmse,max-error,send-rate,drop-rate\n')

        for seed in seeds:
            for drop_rate in [0.0]:  # , 0.05, 0.1]:
                for send_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                    for file, start, path in tests:
                        fce, max_fce, nne, max_nne, lstme, max_lstme = run_test(
                            datafile=file, master_path=path, send_rate=send_rate, drop_rate=drop_rate, seed=seed)
                        if fce is not None:
                            def format_line(method, error, max_error):
                                return '{},{},{},{},{},{}\n'.format(file, method, error, max_error, send_rate, drop_rate)
                            data = format_line('fc', fce[-1], max_fce) + format_line('nn', nne[-1], max_nne) + format_line('lstm', lstme[-1], max_lstme)
                            fp.write(data)
                            fp.flush()


def load_file(datafile, start):
    print("Reading file!")
    data = DataLoader()
    full_path = os.path.join(SRC_DATA_DIR, datafile)
    data.read_file(full_path)
    master_path = data[full_path][PLOT_START:PLOT_END]

    if len(master_path) < PLOT_END:
        return None

    return master_path


def run_test(datafile, master_path, send_rate=0.4, drop_rate=0, seed=1):

    rng = np.random.default_rng(seed)

    trained_model = PredictionModel(send_rate=0.4, report_stats=False, device='cuda')

    # trained_model.load_model('trained_models/2019-10-16-send_rate_0_4_cpu.pb')
    trained_model.load_model('single_01_04_predictor.model')
    trained_model._model.float()

    trained_lstm_model = LSTMPredictionModel(send_rate=0.4, report_stats=False, device='cuda')
    trained_lstm_model.init_model()
    trained_lstm_model.load_model('lstm_predictor.model')

    DeadReckoning_prediction_alg = DRPrediction()
    fc_prediction = PredictionAlgorithm(DeadReckoning_prediction_alg)

    nn_prediction = PredictionAlgorithm(trained_model)
    lstm_prediction = PredictionAlgorithm(trained_lstm_model)

    plt = Plotting()
    _path_state_count = 0
    messages = [master_path[_path_state_count]]
    
    frame = 0
    master_frame = 0
    master_frame_interval = send_rate / DELTA_TIME

    last_message = master_path[master_frame]

    model_predicted_positions = [last_message]
    lstm_model_predicted_positions = [last_message]
    DeadReckoning_predicted_positions = [last_message]

    # store the paths generated for each of the methods
    nn_rcvd_path = [last_message]
    fc_rcvd_path = [last_message]
    lstm_rcvd_path = [last_message]

    while True:
        frame += 1  # next frame
        if frame >= len(master_path):
            # done
            break

        frame_time = frame * DELTA_TIME + master_path[0].get_time()

        # receive a new frame
        if not frame % master_frame_interval:

            messages.append(master_path[frame])

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
        
        predicted_state = nn_prediction.predict(message=nn_rcvd_path[-1],
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

    for (master, fc, nn, lstm) in zip(master_path, DeadReckoning_predicted_positions, model_predicted_positions, lstm_model_predicted_positions):
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
        nn_rmse_ratio = [n / (d + 0.0000001) for n, d in zip(trained_rmse, DeadReckoning_rmse)]

        lstm_total_error += trained_lstm_error[i]
        trained_lstm_rmse.append(sqrt(lstm_total_error / (i+1)))
        lstm_rmse_ratio = [n / (d + 0.0000001) for n, d in zip(trained_lstm_rmse, DeadReckoning_rmse)]

    print(nn_rmse_ratio, lstm_rmse_ratio)
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
        '-d', '--drop-rate', default=0.05, type=float,
        help='Probability of message drop')

    args = parser.parse_args()

    main(args)
#
