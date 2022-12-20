from NN_Model.PredictionModel import PredictionModel
from NN_Model.LSTMPredictionModel import LSTMPredictionModel

import argparse


def main(args):

    if args.lstm:
        print("Training LSTM based model")
        model = LSTMPredictionModel(send_rate=0.4)
    else:
        print("Training fully connected model")
        model = PredictionModel(send_rate=0.4)

    model.using_gpu = args.gpu

    model.train(num_epochs=3, n_batch_size=1000, test_size=60, roll_count=100, seed=35)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--gpu',
        action='store_true',
        help='Use the GPU (nVidia only)')
    argparser.add_argument(
        '--lstm',
        action='store_true',
        help='Train the LSTM model instead of the single state predictive model')
    argparser.add_argument(
        '--stats',
        action='store_true',
        help='Connect to Weights and Biases to log statistics using the current profile')

    args = argparser.parse_args()

    main(args)
