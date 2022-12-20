from util.Vector3 import Vector3
from util.Quaternion import Quaternion

class VehicleState(object):
    def __init__(self):
        self._position = Vector3()
        self._rotation = Quaternion()
        self._velocity = Vector3()
        self._angular_velocity = Vector3()
        self._action_h = 0
        self._action_v = 0
        self._time = -1

    def __str__(self):
        return "{position:" + str(self._position) + \
                ", rotation:" + str(self._rotation) + \
                ", velocity:" + str(self._velocity) + \
                ", angular velocity:" + str(self._angular_velocity) + \
                ", action_h:" + str(self._action_h) + \
                ", action_v:" + str(self._action_v) + \
                ", time:" + str(self._time) + \
                "}"
    def set_position(self, position):
        self._position = Vector3(position)

    def get_position(self):
        return self._position

    def set_rotation(self, rotation):
        if not isinstance(rotation, Quaternion):
            rotation = Quaternion(rotation)
        self._rotation = rotation

    def get_rotation(self):
        return self._rotation

    def set_velocity(self, velocity):
        if not isinstance(velocity, Vector3):
            velocity = Vector3(velocity)
        self._velocity = velocity

    def get_velocity(self):
        return self._velocity

    def set_angular_velocity(self, velocity):
        if not isinstance(velocity, Vector3):
            velocity = Vector3(velocity)
        self._angular_velocity = velocity

    def get_angular_velocity(self):
        return self._angular_velocity

    def set_actions(self, action_h, action_v):
        self._action_h = action_h
        self._action_v = action_v

    def set_time(self, time):
        self._time = time

    def get_time(self):
        return self._time

    def get_action_h(self):
        return self._action_h

    def get_action_v(self):
        return self._action_v
