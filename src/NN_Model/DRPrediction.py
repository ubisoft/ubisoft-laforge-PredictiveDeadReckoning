from util.VehicleState import VehicleState


class DRPrediction:
    @staticmethod
    def predict(message, time_to_predict, delta_time):
        message_pos = message.get_position()
        message_vel = message.get_velocity()
        message_ang_vel = message.get_angular_velocity()

        num_intervals = int(time_to_predict/delta_time + 0.5)
        rotation_each_interval = message_ang_vel[1]*delta_time

        predicted_position = message_pos
        rotated_velocity = message_vel
        for i in range(num_intervals):
            rotated_velocity = rotated_velocity.rotation_with_angle(rotation_each_interval, "y")
            predicted_position += rotated_velocity*delta_time
            predicted_state = VehicleState()
            predicted_state.set_time(message.get_time() + time_to_predict)
            predicted_state.set_position(predicted_position)
            predicted_state.set_rotation(message.get_rotation())
            predicted_state.set_velocity(rotated_velocity)
            predicted_state.set_angular_velocity(message_ang_vel)
        return predicted_state

    def predict_path(self, message, time_to_predict, delta_time):
        predicted_path = []
        message_pos = message.get_position()
        # predicted_path.append(message)
        message_vel = message.get_velocity()
        message_ang_vel = message.get_angular_velocity()

        num_intervals = int(time_to_predict/delta_time + 0.5)
        rotation_each_interval = message_ang_vel[1]*delta_time

        predicted_position = message_pos
        rotated_velocity = message_vel
        for i in range(num_intervals):
            rotated_velocity = rotated_velocity.rotation_with_angle(rotation_each_interval, "y")
            predicted_position += rotated_velocity*delta_time
            predicted_state = VehicleState()
            predicted_state.set_time(message.get_time() + i*delta_time)
            predicted_state.set_position(predicted_position)
            predicted_state.set_rotation(message.get_rotation())
            predicted_state.set_velocity(rotated_velocity)
            predicted_state.set_angular_velocity(message_ang_vel)
            predicted_path.append(predicted_state)
        return predicted_path


class DRBlending:

    def clamp(x, minval, maxval):
        return max(minval, min(x, maxval))

    @staticmethod
    def blend(current_state, predicted_state, delta_time):
        current_pos = current_state.get_position()
        current_vel = current_state.get_velocity()
        predicted_pos = predicted_state.get_position()
        dx = predicted_pos - current_pos
        v_prime = dx*(1/0.4)
        delta_v = v_prime - current_vel
        a = delta_v*(1.0/delta_time)
        # a = a.clamp(0,25)
        v = current_vel + a*delta_time
        # v = v.clamp(0,30)
        smooth_state = VehicleState()
        smooth_state.set_time(current_state.get_time() + delta_time)
        smooth_state.set_position(current_pos + v*delta_time)
        smooth_state.set_rotation(current_state.get_rotation())
        smooth_state.set_velocity(v)
        smooth_state.set_angular_velocity(current_state.get_angular_velocity())
        return smooth_state
