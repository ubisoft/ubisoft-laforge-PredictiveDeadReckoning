from util.VehicleState import VehicleState
from util.Vector3 import Vector3
from math import sin, cos, pi, isclose

class PathTrackingBlending:    
    @staticmethod
    def blend(current_state, predicted_state, delta_time, gamma1 = 50.0, gamma2 = 10.0, gamma3 = 0.2):
        current_pos = current_state.get_position()
        current_vel = current_state.get_velocity()
        predicted_pos = predicted_state.get_position()
        predicted_vel = predicted_state.get_velocity()
        e_vec = predicted_pos - current_pos
        e_H = predicted_vel.angle_about_axis(current_vel, "y")
        if (predicted_vel*current_vel)[1]<0:
            e_H = -e_H
        e_H = max(-pi/2+0.05, min(e_H, pi/2-0.05))
        #print(e_H)
        e_X = e_vec.dot(predicted_vel.normalized())*predicted_vel.normalized()
        e_Lvec = e_vec - e_vec.dot(predicted_vel.normalized())*predicted_vel.normalized()
        e_L = e_Lvec.magnitude()
        if isclose((e_X.normalized() + predicted_vel.normalized()).magnitude(), 0.0):
            v = current_vel*0.5
        else:
            v = current_vel
        if (e_Lvec*predicted_vel)[1]<0:
            e_L = -e_L
        #print(e_L)
        if isclose(v.magnitude(), 0):
            v = predicted_vel
            rotation = 0
        else:
            v = current_vel.normalized()*e_vec.magnitude()/gamma3
            #print(v)
            rotation = -(-gamma1*e_L-gamma2*v.magnitude()*sin(e_H))/(v.magnitude()*cos(e_H))
            rotation = rotation*0.02
            #print(rotation*180.0/pi)
            v = v.rotation_with_angle(rotation,"y")
        v = Vector3(vec=[v[0], 0, v[2]])
        smooth_state = VehicleState()
        smooth_state.set_time(current_state.get_time() + delta_time)
        smooth_state.set_position(current_pos + v*delta_time)
        smooth_state.set_rotation(current_state.get_rotation())
        smooth_state.set_velocity(v)
        smooth_state.set_angular_velocity(rotation)
        return smooth_state