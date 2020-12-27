import numpy as np


class PurePursuitLaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    """

    def __init__(self, parameters):
        self.v_max = parameters["v_max"]
        self.k = parameters["k"]
        self.look_ahead = parameters["look_ahead"]

    def pure_pursuit(self, d, phi, last_v, last_w, vehicle_msg):
        print(vehicle_msg)
        if d is None or phi is None or np.isnan(d) or np.isnan(phi) or d >= self.look_ahead:
            v = last_v
            w = last_w
            return v, w
        else:
            x = np.arcsin(d / self.look_ahead)
            alpha = - x - phi
            sin_alpha = np.sin(alpha)

            w = sin_alpha * self.k
            v = self.v_max
            return v, w
