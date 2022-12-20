#!/usr/bin/env python

import sys
from DataLoader.DataLoader import DataLoader
from NN_Model.PredictionModel import PredictionModel
from NN_Model.BlendingModel import BlendingModel
from time import time
from plot.plot_path import Plotting
from NN_Model.PredictionAlgorithm import PredictionAlgorithm, BlendingAlgorithm
from NN_Model.DRPrediction import DRPrediction, DRBlending
from util.VehicleState import VehicleState
from generate_blending_data.SOCP_solver import SOCP
from util.Vector3 import Vector3
from random import random
import matplotlib
from math import sqrt
import argparse


def clamp_dvel(state1, state2, args):
    velocity = state2.get_velocity()
    delta_vel_mag = (state2.get_velocity() - state1.get_velocity()).magnitude()/args.delta_time
    if (delta_vel_mag > args.dvel_max):
        velocity = velocity * args.dvel_max/delta_vel_mag
    return state1.get_position() + args.delta_time*velocity, velocity


def SOCP_blending(args, path_to_blend, path, current_state=None):

    if current_state == None:
        _current_state = VehicleState()
        _current_state.set_position(path[-1].get_position())
        _current_state.set_velocity(path[-1].get_velocity())
        _current_state.set_time(path[-1].get_time())
    else:
        _current_state = current_state

    problem = SOCP(path_to_blend, _current_state)
    result_v_x, result_v_y, result_p_x, result_p_y = problem.solve()
    for _k in range(len(path_to_blend)):
        SOCP_vel = Vector3(vec=[result_v_x[_k].value, 0.0, result_v_y[_k].value])
        SOCP_state = VehicleState()
        SOCP_state.set_time(path[-1].get_time() + args.delta_time)
        # SOCP_state.set_position(path[-1].get_position()+SOCP_vel*args.delta_time)
        SOCP_state.set_position(Vector3(vec=[result_p_x[_k].value, 0.0, result_p_y[_k].value]))
        SOCP_state.set_velocity(SOCP_vel)
        path.append(SOCP_state)


def ML_blending(alg, args, path_to_blend, path, current_state=None):
    if current_state == None:
        _current_state = VehicleState()
        _current_state.set_position(path[-1].get_position())
        _current_state.set_velocity(path[-1].get_velocity())
        _current_state.set_time(path[-1].get_time())
    else:
        _current_state = current_state

    for _k in range(len(path_to_blend) - 1):
        blended_state = alg.blend(current_state=_current_state,
                                  predicted_state=path_to_blend[-1],
                                  blend_time=path_to_blend[-1].get_time()-_current_state.get_time(),
                                  delta_time=args.delta_time)

        _pos, _vel = clamp_dvel(
            _current_state, blended_state, args
        )
        blended_state.set_position(_pos)
        blended_state.set_velocity(_vel)

        _current_state = blended_state
        path.append(blended_state)


def deadreckoning_blend(alg, args, path_to_blend, path):
    _current_state = path[-1]
    for _k in range(len(path_to_blend)-1):
        _state = alg.blend(
            current_state=_current_state,
            predicted_state=path_to_blend[_k+1],
            delta_time=args.delta_time
        )
        path.append(_state)
        _current_state = _state


def time_between_states(path, state1, state2):
    return path[state2].get_time() - path[state1].get_time()


def is_send_message(args, state, prev_msg_state, path):
    time_between_states = path[state].get_time() - path[prev_msg_state].get_time()
    if time_between_states < args.send_rate:
        return False
    return True


def blend(args, master_path):
    # ---------------------------------------------------------
    # Set initial state of the trajectories
    # ---------------------------------------------------------
    messages_sent = [
        master_path[0]
    ]
    model_blending_trajectory = [
        master_path[0]
    ]
    SOCP_blending_trajectory = [
        master_path[0]
    ]
    DeadReckoning_blending_trajectory = [
        master_path[0]
    ]
    DeadReckoning_predicted_states = [
        master_path[0]
    ]

    # ---------------------------------------------------------
    # Load blending trained model
    # ---------------------------------------------------------
    trained_blending_model = BlendingModel()
    trained_blending_model.init_model()
    trained_blending_model.load_model('trained_models/blending_model-21_10_2019_13_57_26_cpu.pb')
    model_blending = BlendingAlgorithm(trained_blending_model)

    # ---------------------------------------------------------
    # Load DeadReckoning prediction and blending algorithm
    # ---------------------------------------------------------
    DeadReckoning_prediction_alg = DRPrediction()
    DeadReckoning_prediction = PredictionAlgorithm(DeadReckoning_prediction_alg)
    DeadReckoning_blending = DRBlending()

    _path_state_count = 0
    _prev_message_state = 0
    _last_message = messages_sent[-1]
    _new_message = True
    while _path_state_count < len(master_path) - 1:
        if is_send_message(args, _path_state_count, _prev_message_state, master_path):
            _new_message = True
            _prev_message_state = _path_state_count
            messages_sent.append(
                master_path[_prev_message_state]
            )
            _last_message = messages_sent[-1]

        if not _new_message:
            _path_state_count += 1
            continue

        _path_state_count += 1
        _new_message = False

        _partial_path_count = _prev_message_state + 1
        if _partial_path_count > len(master_path) - 1:
            break
        # ---------------------------------------------------------
        # Get the predicted states between to messages
        # ---------------------------------------------------------
        _partial_estimated_path = []

        _time_between_states = time_between_states(
            master_path, _prev_message_state, _partial_path_count
        )

        while (_time_between_states < args.send_rate):
            _predicted_state = DeadReckoning_prediction.predict(
                message=_last_message,
                time_to_predict=_time_between_states,
                delta_time=args.delta_time
            )
            _partial_estimated_path.append(_predicted_state)
            DeadReckoning_predicted_states.append(_predicted_state)
            _partial_path_count += 1

            if _partial_path_count > len(master_path) - 1:
                break

            _time_between_states = time_between_states(
                master_path, _prev_message_state, _partial_path_count
            )

        # ---------------------------------------------------------
        # DeadReckoning blending
        # ---------------------------------------------------------
        deadreckoning_blend(
            alg=DeadReckoning_blending,
            args=args,
            path_to_blend=_partial_estimated_path,
            path=DeadReckoning_blending_trajectory)

        # ---------------------------------------------------------
        # ML blending
        # ---------------------------------------------------------
        ML_blending(
            alg=model_blending,
            args=args,
            path_to_blend=_partial_estimated_path,
            path=model_blending_trajectory,
            current_state=None
        )

        # ---------------------------------------------------------
        # SOCP blending
        # ---------------------------------------------------------
        # SOCP_blending(
        #     args = args,
        #     path_to_blend = _partial_estimated_path,
        #     path = SOCP_blending_trajectory,
        #     current_state=_last_message
        # )

    return {
        'messages': messages_sent,
        'predictions': DeadReckoning_predicted_states,
        'deadreckoning_trajectory': DeadReckoning_blending_trajectory,
        'ml_trajectory': model_blending_trajectory,
        'SOCP_trajectory': SOCP_blending_trajectory
    }


def main(arg):

    # ---------------------------------------------------------
    # Parse arguments
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Blending algorithm arguments")
    parser.add_argument(
        'file_name', metavar="f",
        default="2019-09-20-13-17-45_MasterCar.txt",
        action='store', nargs='?'
    )
    parser.add_argument(
        'send_rate', metavar='s', default=0.5, action='store', nargs='?'
    )
    parser.add_argument(
        'frame_rate', metavar='fr', default=0.02, action='store', nargs='?'
    )
    parser.add_argument(
        'plotting_start', metavar='pls', default=500, action='store', nargs='?'
    )
    parser.add_argument(
        'plotting_end', metavar='ple', default=1000, action='store', nargs='?'
    )
    parser.add_argument(
        'path_start', metavar='ps', default=500, action='store', nargs='?'
    )
    parser.add_argument(
        'path_end', metavar='pe', default=1000, action='store', nargs='?'
    )
    parser.add_argument(
        'dvel_max', metavar='dvel', default=1000, action='store', nargs='?'
    )
    parser.add_argument(
        'delta_time', metavar='dt', default=0.02, action='store', nargs='?'
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load master vehicle's trajectory
    # ---------------------------------------------------------
    data = DataLoader()
    data.read_file(file_name="training-data/" + args.file_name)

    master_path = data["training-data/" + args.file_name]
    master_path = master_path[args.path_start:args.path_end]

    # ---------------------------------------------------------
    # Blend
    # ---------------------------------------------------------
    res = blend(args, master_path)

    # ---------------------------------------------------------
    # PLotting the results
    # ---------------------------------------------------------
    plt = Plotting()
    # plot the master path
    plt.plot_path(master_path, color='r')

    # plot predicted states
    plt.plot_predictions(res['predictions'], style="*", markersize=5.0)

    # plot messages
    plt.plot_predictions(res['messages'],  style="o", markersize=7.0)

    # plot deadreckoning blending trajectory
    plt.plot_path(path=res['deadreckoning_trajectory'], color='g', style="-")

    # plot ML blending trajectory
    plt.plot_path(path=res['ml_trajectory'], color='b', style="-")

    # plot SCOP blending trajectory
    # plt.plot_path(path=res['SOCP_trajectory'], color='k', style="-")

    plt.show(equal=True)


if __name__ == "__main__":
    main(sys.argv)
#
