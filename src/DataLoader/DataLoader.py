from util.VehicleState import VehicleState
from os import listdir
from tqdm import tqdm

class Network:
    def __init__(self):
        self._send_rate     = 0
        self._latency       = 0
        self._packet_loss   = 0

class DataLoader(Network):
    def __init__(self, folder_name = None, file_name = None, replica_file = False):
        self.folder_name    = folder_name
        self.file_name      = file_name
        self.replica_file   = replica_file
        self._data = {}

    # TODO: remove tetra dataset
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= len(self._data):
            return
        else:
            raise StopIteration

    def __str__(self):
        return str(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, x):
        return self._data[x]

    def read_all_files_in_folder(self, folder_name=None, replica_files = None):

        if (folder_name == None):
            folder_name = self.folder_name
        
        if folder_name == 'collision-training-data/':
            for file in tqdm(listdir('data/' + folder_name)):
                if file.endswith("_MasterCar.txt"):
                    self.read_collision_file('data/' + folder_name + file)
        else:
            for file in tqdm(listdir(folder_name)):
                if file.endswith(".txt"):
                    file_name = file.split(".")[0]
                    file_name = file_name.split("_")
                    if (replica_files == None):
                        self.read_file(
                            file_name = folder_name + file_name[0] + "_MasterCar.txt"
                        )
                    if (replica_files == "ML"):
                        self.read_file(
                            file_name = folder_name + file_name[0] + "_MLReplicaCar.txt"
                        )


    def read_file(self, file_name = None):
        if file_name == None:
            file_name = self.file_name

        self._data[file_name] = []
        try:
            file = open(file_name)
            for line in file:
                l = line.split("|")
                if l[0] == 'send-rate':
                    self._send_rate = float(l[1])
                if l[0] == 'latency':
                    self._latency = float(l[1])

                if l[0] == 'packet-drop':
                    self._packet_loss = float(l[1])
                if len(l) >= 7:
                    state = VehicleState()
                    state.set_time(float(l[0]))
                    state.set_position(self.parse_vec(l[1]))
                    state.set_rotation(self.parse_vec(l[2]))
                    state.set_velocity(self.parse_vec(l[3]))
                    state.set_angular_velocity(self.parse_vec(l[4]))
                    state.set_actions(float(l[5]), float(l[6]))
                    self._data[file_name].append(state)
        except Exception as e:
            print("Error in reading file: ", file_name, e)
            return

    def read_collision_file(self, file_name = None):
        if file_name == None:
            file_name = self.file_name
        self._data[file_name] = []
        try:
            file = open(file_name).readlines()
            prev_collision = False
            collision_data = [9999.0];
            for num, line in enumerate(file[:-1]):
                l = line.split("|")
                if len(file[num+1].split("|")) == 11:
                    if prev_collision == False:
                        i = 1
                        if(num+i>=len(file)-1):
                            break
                        l_next = file[num+i].split("|")
                        collision_time = float(l[0])
                        data = [0.0]
                        data = data + self.parse_vec(l[1])
                        data = data + self.parse_vec(l[2])
                        data = data + self.parse_vec(l[3])
                        data = data + self.parse_vec(l[4])
                        data.append(float(l[5]))
                        data.append(float(l[6]))
                        data = data + self.parse_vec(l_next[8])
                        data = data + self.parse_vec(l_next[9])
                        data.append(float(l_next[10]))
                        collision_data = list(data)
                        self._data[file_name].append(data)
                        while float(l_next[0]) - collision_time < 1.02:
                            data = [float(l_next[0]) - collision_time]
                            data = data + self.parse_vec(l_next[1])
                            data = data + self.parse_vec(l_next[2])
                            data = data + self.parse_vec(l_next[3])
                            data = data + self.parse_vec(l_next[4])
                            data.append(float(l[5]))
                            data.append(float(l[6]))
                            data = data + collision_data[14:]
                            self._data[file_name].append(data)
                            i = i + 1
                            if(num+i>=len(file)-1):
                                break
                            l_next = file[num+i].split("|")
                    prev_collision = True
                else:
                    prev_collision = False
        except Exception as e:
            print("Error in reading file: ", file_name, e)
            return

    def read_sliding_from_collision_file(self, file_name = None):
        if file_name == None:
            file_name = self.file_name
        self._data[file_name] = []
        try:
            file = open(file_name).readlines()
            prev_collision = False
            for num, line in enumerate(file[:-1]):
                l = line.split("|")
                if len(file[num+1].split("|")) == 11:
                    if prev_collision == False:
                        i = 1
                        if(num+i>=len(file)-1):
                            break
                        l_next = file[num+i].split("|")
                        collision_time = float(l[0])
                        while float(l_next[0]) - collision_time < 2.0:
                            if float(l_next[0]) - collision_time > 0.5:
                                state = VehicleState()
                                state.set_time(float(l[0]))
                                state.set_position(self.parse_vec(l[1]))
                                state.set_rotation(self.parse_vec(l[2]))
                                state.set_velocity(self.parse_vec(l[3]))
                                state.set_angular_velocity(self.parse_vec(l[4]))
                                state.set_actions(float(l[5]), float(l[6]))
                                self._data[file_name].append(state)
                            i = i + 1
                            if(num+i>=len(file)-1):
                                break
                            l_next = file[num+i].split("|")
                    prev_collision = True
                else:
                    prev_collision = False
        except Exception as e:
            print("Error in reading file: ", file_name, e)
            return

    def read_blending_file(self, file_name = None):
        if file_name == None:
            file_name = self.file_name
        self._data[file_name] = []
        try:
            file = open(file_name)
            for line in file:
                l = line.split("|")
                if len(l) == 5:
                    data = []
                    data = data + self.parse_vec(l[0])
                    data = data + self.parse_vec(l[1])
                    data = data + self.parse_vec(l[2])
                    data.append(float(l[3]))
                    data = data + self.parse_vec(l[4])
                    self._data[file_name].append(data)
        except Exception as e:
            print("Error in reading file: ", file_name, e)
            return
        
    @staticmethod
    def parse_vec(vec):
        vector = []
        vec = vec.split("(")[1]
        vec = vec.split(")")[0]
        vec = vec.split(",")
        for _i in range(len(vec)):
            vector.append(round(float(vec[_i]), 5))
        return vector

    def read_message_file(self, file_name = None):
        messages = []
        if file_name == None:
            file_name = self.file_name.split(".")[0] + ".mg"

        file = open(file_name)
        for line in file:
            l  = line.split("|")
            mg = Message()
            mg.set_message_time(float(l[0]))
            s  = VehicleState()
            s.set_position(self.parse_vec(l[2]))
            s.set_rotation(self.parse_vec(l[3]))
            s.set_velocity(self.parse_vec(l[4]))
            s.set_angular_velocity(self.parse_vec(l[4]))
            mg.set_message_state(s)
            messages.append(mg)
        return messages
