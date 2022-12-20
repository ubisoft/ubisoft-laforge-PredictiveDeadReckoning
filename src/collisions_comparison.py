from plot.plot_path import Plotting
from DataLoader.DataLoader import DataLoader
from NN_Model.DRPrediction import DRPrediction
from statistics import mean, stdev
from math import pi
import matplotlib.pyplot as mp
import matplotlib.patches as mpatches
import numpy as np
from copy import deepcopy

FILE_NAME_BASE = 'collected-collision-data/2020-05-25-17-53-26'  # messages forced to be sent after 250ms (sliding network for 2 seconds) 250 ms send rate

TIME_AFTER_COLLISION = 2.5
collision_count = 0
num_teleports = 0

MASTER_FILE = FILE_NAME_BASE + '_MasterCar.txt'
TETRA_FILE = FILE_NAME_BASE + '_TetraReplicaCar.txt'
ML_FILE = FILE_NAME_BASE + '_MLReplicaCar.txt'
PVB_FILE = FILE_NAME_BASE + '_PVBReplicaCar.txt'
send_rate = 0.5
data = DataLoader()
data.read_file(MASTER_FILE)
data.read_file(TETRA_FILE)
data.read_file(ML_FILE)
# data.read_file(PVB_FILE)
interval = int(send_rate/0.02)
newCollision = False
x = np.arange(0, TIME_AFTER_COLLISION, 0.02)
ML_pos_error = []
ML_rot_error = []
base_pos_error = []
base_rot_error = []

collision_time = 0.0
prev_state = data[MASTER_FILE][0]
maxidx = min(len(data[MASTER_FILE]), len(data[TETRA_FILE]), len(data[ML_FILE]))
for index in range(maxidx):
    state = data[MASTER_FILE][index]
    #print(abs(prev_state.get_velocity().angle_about_axis(state.get_velocity(), "y")))
    if (state.get_position() - prev_state.get_position()).magnitude() > 5.0:
        newCollision = True
        num_teleports += 1
    elif newCollision and abs(prev_state.get_velocity().angle_about_axis(state.get_velocity(), "y")) > 0.3:
        newCollision = False
        collision_time = state.get_time()
#        print(collision_time)
        collision_count += 1
    if state.get_time() - collision_time < TIME_AFTER_COLLISION-0.0001:
        bucket = round((state.get_time() - collision_time)/0.02)
        ML_state = data[ML_FILE][index]
        Tetra_state = data[TETRA_FILE][index]
        if bucket >= len(ML_pos_error):
            ML_pos_error.append([(state.get_position()-ML_state.get_position()).magnitude()])
            ML_rot_error.append([min(abs(state.get_rotation().angle(ML_state.get_rotation())*180/pi-360),
                                state.get_rotation().angle(ML_state.get_rotation())*180/pi)])
            base_pos_error.append([(state.get_position()-Tetra_state.get_position()).magnitude()])
            base_rot_error.append([min(abs(state.get_rotation().angle(Tetra_state.get_rotation())*180/pi-360),
                                  state.get_rotation().angle(Tetra_state.get_rotation())*180/pi)])
        else:
            ML_pos_error[bucket].append((state.get_position()-ML_state.get_position()).magnitude())
            ML_rot_error[bucket].append(min(abs(state.get_rotation().angle(ML_state.get_rotation())*180/pi-360),
                                        state.get_rotation().angle(ML_state.get_rotation())*180/pi))
            base_pos_error[bucket].append((state.get_position()-Tetra_state.get_position()).magnitude())
            base_rot_error[bucket].append(min(abs(state.get_rotation().angle(Tetra_state.get_rotation())*180/pi-360),
                                          state.get_rotation().angle(Tetra_state.get_rotation())*180/pi))

    prev_state = deepcopy(state)

print(collision_count)
print(num_teleports)

means = []
upper = []
lower = []
for y in ML_pos_error:
    stddev = stdev(y)
    means.append(mean(y))
    upper.append(mean(y) + 2*stddev)
    lower.append(mean(y) - 2*stddev)
mp.figure()
mp.title('Position Error After Collision')
mp.plot(x, means)
mp.fill_between(x, lower, upper, alpha=0.2)
means = []
upper = []
lower = []
for y in base_pos_error:
    stddev = stdev(y)
    means.append(mean(y))
    upper.append(mean(y) + 2*stddev)
    lower.append(mean(y) - 2*stddev)
mp.plot(x, means)
mp.fill_between(x, lower, upper, alpha=0.2)
mp.xlabel('time to predict (s)')
mp.ylabel('error (m)')
mp.legend(['ML', 'Tetra'])

means = []
upper = []
lower = []
for y in ML_rot_error:
    stddev = stdev(y)
    means.append(mean(y))
    upper.append(mean(y) + 2*stddev)
    lower.append(mean(y) - 2*stddev)
mp.figure()
mp.title('Rotation Error After Collision')
mp.plot(x, means)
mp.fill_between(x, lower, upper, alpha=0.2)
means = []
upper = []
lower = []
for y in base_rot_error:
    stddev = stdev(y)
    means.append(mean(y))
    upper.append(mean(y) + 2*stddev)
    lower.append(mean(y) - 2*stddev)
mp.plot(x, means)
mp.fill_between(x, lower, upper, alpha=0.2)
mp.xlabel('time to predict (s)')
mp.ylabel('error (deg)')
mp.legend(['ML', 'Tetra'])


mp.figure()
mp.title('Position Error After Collision')
mp.violinplot([i for sublist in ML_pos_error for i in sublist])
mp.ylabel('error (m)')
mp.violinplot([i for sublist in base_pos_error for i in sublist])
mp.legend(['ML', 'Tetra'])

mp.figure()
mp.title('Rotation Error After Collision')
mp.violinplot([i for sublist in ML_rot_error for i in sublist])
mp.ylabel('error (deg)')
mp.violinplot([i for sublist in base_rot_error for i in sublist])
mp.legend(['ML', 'Tetra'])

mp.show()
