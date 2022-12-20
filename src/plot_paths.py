from plot.plot_path import Plotting
from DataLoader.DataLoader import DataLoader
from NN_Model.DRPrediction import DRPrediction
from statistics import mean
import matplotlib.pyplot as mp
import matplotlib.patches as mpatches

# FILE_NAME_BASE = 'training-data/2019-10-23-16-37-26' # 300ms
# FILE_NAME_BASE = 'training-data/2019-10-23-16-54-01' # 100ms
# FILE_NAME_BASE = 'training-data/2019-10-23-16-55-57' # 500ms
# FILE_NAME_BASE = 'training-data/2019-11-04-15-28-47' # ML
#FILE_NAME_BASE = 'training-data/2019-11-19-14-02-24'
# FILE_NAME_BASE = 'training-data/2019-12-12-18-01-01' # 300ms, 0.2 drop rate
# FILE_NAME_BASE = 'training-data/2019-12-12-17-58-41' # 200ms, 0.2 drop rate
# FILE_NAME_BASE = 'training-data/2019-12-12-18-18-11' # 200ms, 0.0 drop rate
# FILE_NAME_BASE = 'training-data/2019-12-12-18-30-05' # 300ms, 0.5 drop rate
# FILE_NAME_BASE = 'training-data/2019-12-12-19-14-54' # 300ms, 0.5 drop rate ML CAR
# FILE_NAME_BASE = 'collected-collision-data/2020-05-25-16-43-47' # collision
# FILE_NAME_BASE = 'collected-collision-data/2020-05-25-18-20-39' # better collision
# FILE_NAME_BASE = 'collected-collision-data/2020-05-25-18-39-55' # better collision #message after 500ms
# FILE_NAME_BASE = 'collected-collision-data/2020-05-25-18-42-04' # better collision #message after 500ms
FILE_NAME_BASE = 'collected-collision-data/2020-05-26-10-47-12'  # better collision #message after 500ms


MASTER_FILE = FILE_NAME_BASE + '_MasterCar.txt'
TETRA_FILE = FILE_NAME_BASE + '_TetraReplicaCar.txt'
ML_FILE = FILE_NAME_BASE + '_MLReplicaCar.txt'
#PVB_FILE = FILE_NAME_BASE + '_PVBReplicaCar.txt'
send_rate = 0.2
data = DataLoader()
data.read_file(MASTER_FILE)
data.read_file(TETRA_FILE)
data.read_file(ML_FILE)
# data.read_file(PVB_FILE)
interval = int(send_rate/0.02)
predictions = []
predictions.append(data[MASTER_FILE][0])
predictor = DRPrediction()
tetra_errors = []
adjusted_tetra_errors = []
pt_errors = []
adjusted_pt_errors = []
#PVB_errors = []
#adjusted_PVB_errors = []

for i in range(1, min(len(data[TETRA_FILE]), len(data[ML_FILE]), len(data[MASTER_FILE]))):
    # print(i)
    msg_index = i - i % interval
    if i % interval == 0:
        predictions.append(data[MASTER_FILE][i])
    else:
        predictions.append(predictor.predict(message=data[MASTER_FILE][msg_index], time_to_predict=i % interval*0.02, delta_time=0.02))
    tetra_errors.append((data[TETRA_FILE][i].get_position()-data[MASTER_FILE][i].get_position()).magnitude())
    pt_errors.append((data[ML_FILE][i].get_position()-data[MASTER_FILE][i].get_position()).magnitude())
#    PVB_errors.append((data[PVB_FILE][i].get_position()-data[MASTER_FILE][i].get_position()).magnitude())
#    adjusted_PVB_errors.append((data[PVB_FILE][i].get_position()-data[MASTER_FILE][i].get_position()).magnitude())
    if i >= int(0.4/0.02):
        adjusted_tetra_errors.append((data[TETRA_FILE][i].get_position()-data[MASTER_FILE][i-int(0.4/0.02)].get_position()).magnitude())
    if i >= int(0.08/0.02):
        adjusted_pt_errors.append((data[ML_FILE][i].get_position()-data[MASTER_FILE][i-int(0.1/0.02)].get_position()).magnitude())
#        adjusted_PVB_errors.append((data[PVB_FILE][i].get_position()-data[MASTER_FILE][i-int(0.08/0.02)].get_position()).magnitude())


print('Tetra: ' + str(mean(tetra_errors)))
print('Adjusted Tetra: ' + str(mean(adjusted_tetra_errors)))
print('PT: ' + str(mean(pt_errors)))
print('Adjusted PT: ' + str(mean(adjusted_pt_errors)))

plt = Plotting()
plt.plot_path_orientation(path=data[MASTER_FILE][15300:15500], color='r', label='Master')
#plt.plot_path(path = predictions, color='c', style = '--', label = 'Predictions')
plt.plot_path_orientation(path=data[TETRA_FILE][15300:15500], color='g', label='Tetra')
#plt.plot_path(path = data[PVB_FILE], color='b', style = '--', label = 'Path Tracking')
plt.plot_path_orientation(path=data[ML_FILE][15300:15500], color='m', label='Collision Network')
plt.show(equal=True)

#plt = Plotting()
#plt.plot_path(path = data[MASTER_FILE], color='r', label = 'Master')
##plt.plot_path(path = predictions, color='c', style = '--', label = 'Predictions')
#plt.plot_path(path = data[TETRA_FILE], color='g', label = 'Tetra')
##plt.plot_path(path = data[PVB_FILE], color='b', style = '--', label = 'Path Tracking')
#plt.plot_path(path = data[ML_FILE], color='m', label = 'Collision Network')
# plt.show(equal=True)

#data = [tetra_errors,PVB_errors,pt_errors]
#adjusted_data = [adjusted_tetra_errors[200:],adjusted_PVB_errors[200:],adjusted_pt_errors[200:]]
# mp.figure()
#FC_patch = mpatches.Patch(color='g', label='DeadReckoning')
#PVB_patch = mpatches.Patch(color='b', label='Path Tracking')
#PT_patch = mpatches.Patch(color='m', label='ML + Path Tracking')
# mp.legend(handles=[FC_patch,PVB_patch,PT_patch])
#mp.title('0.3 send rate, 0.5 drop rate')
#plot = mp.violinplot(adjusted_data, showmeans=True)
# plot['bodies'][0].set_facecolor('g')
# plot['bodies'][1].set_facecolor('b')
# plot['bodies'][2].set_facecolor('m')
# plot['bodies'][0].set_edgecolor('g')
# plot['bodies'][1].set_edgecolor('b')
# plot['bodies'][2].set_edgecolor('m')
