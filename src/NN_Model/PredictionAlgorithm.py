class PredictionAlgorithm:
    def __init__(self, predictor = None):
        self._predictor = predictor
    
    def predict(self, message, time_to_predict, delta_time):
        if self._predictor == None:
            return None

        return self._predictor.predict(message, time_to_predict, delta_time)
    
    def predict_path(self, message, time_to_predict, delta_time):
        if self._predictor == None:
            return None

        return self._predictor.predict_path(message, time_to_predict, delta_time)


class BlendingAlgorithm:
    def __init__(self, blender = None):
        self._blender = blender
    
    def blend(self, current_state, predicted_state, blend_time, delta_time):
        if self._blender == None:
            return None

        return self._blender.blend(current_state, predicted_state, blend_time, delta_time)
    
