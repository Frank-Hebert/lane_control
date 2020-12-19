import numpy as np


class PurePursuitLaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    """

    def __init__(self, parameters):

        self.parameters = parameters

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters

    # def pure_pursuit(self, d, phi, last_v, last_w):
    #
    #     v_max = 0.2
    #     # x =
    #
    #     K = 0.2
    #     L = 0.5
    #     # v_r = 0.5
    #     # L = K * v_r
    #     print(d)
    #     if d == None or phi == None or np.isnan(d) or np.isnan(phi) or d >= L:
    #         v = last_v
    #         w = last_w
    #         return v, w
    #     else:
    #
    #         x = np.arcsin(d / L)
    #         alpha = -x - phi
    #         sin_alpha = np.sin(alpha)
    #
    #         w = sin_alpha / K
    #
    #         v = v_max * np.cos(alpha)
    #         return v, w

    # def pure_pursuit(self, d, phi, last_v, last_w):

    #     v_max = 0.2
    #     # x =

    #     K = 0.2
    #     L = 0.4
    #     # v_r = 0.5
    #     # L = K * v_r
    #     if d == None or phi == None or np.isnan(d) or np.isnan(phi) or d >= L:
    #         v = last_v
    #         w = last_w
    #         return v, w
    #     else:

    #         x = np.arcsin(d / L)
    #         alpha = -x - phi
    #         # alpha = -phi
    #         print(f"x : {x},      phi : {phi},       alpha : {alpha}.")
    #         sin_alpha = np.sin(alpha)

    #         w = sin_alpha / K
    #         if np.cos(alpha) >= 0.95:
    #             v = v_max + 0.5
    #         else:
    #             v = v_max * np.cos(alpha)
    #         return v, w

    # def pure_pursuit(self, d, phi, last_v, last_w):

    #     v_max = 0.2
    #     # x =

    #     K = 0.2
    #     L = 0.4
    #     # v_r = 0.5
    #     # L = K * v_r
    #     if d == None or phi == None or np.isnan(d) or np.isnan(phi) or d >= L:
    #         v = last_v
    #         w = last_w
    #         return v, w
    #     else:

    #         x = np.arcsin(d / L)
    #         alpha = -x - phi
    #         # alpha = -phi
    #         print(f"x : {x},      phi : {phi},       alpha : {alpha}.")
    #         sin_alpha = np.sin(alpha)

    #         w = sin_alpha / K
    #         if abs(np.cos(phi)) >= 0.95:
    #             v = v_max + 0.3
    #         else:
    #             v = v_max * np.cos(alpha)
    #         return v, w


    # def pure_pursuit(self, d, phi, last_v, last_w):
    #
    #     v_max = 0.4
    #     # x =
    #
    #     K = 5.0
    #     L = 0.2
    #     # v_r = 0.5
    #     # L = K * v_r
    #     print(d)
    #     if d == None or phi == None or np.isnan(d) or np.isnan(phi) or d >= L:
    #         v = last_v
    #         w = last_w
    #         return v, w
    #     else:
    #         if phi == 0:
    #             alpha = np.arcsin(d / L)
    #
    #         else:
    #             x = np.arcsin(d / L)
    #             alpha = -x - phi
    #
    #         sin_alpha = np.sin(alpha)
    #
    #         w = K * sin_alpha
    #
    #         v = v_max * np.cos(alpha)
    #         return v, w


    # def pure_pursuit(self, d, phi, last_v, last_w):
    #
    #     v_max = 0.3
    #     # x =
    #
    #     K = 5
    #     L = 0.25
    #     # v_r = 0.5
    #     # L = K * v_r
    #     if d == None or phi == None or np.isnan(d) or np.isnan(phi) or d >= L:
    #         v = last_v
    #         w = last_w
    #         return v, w
    #     else:
    #         if phi == 0:
    #             alpha = np.arcsin(d / L)
    #
    #         else:
    #             x = np.arcsin(d / L)
    #             alpha = -x - phi
    #         sin_alpha = np.sin(alpha)
    #
    #         w = K * sin_alpha
    #         if np.abs(alpha) <= 0.3:
    #             v = v_max + 0.1
    #         else:
    #             # v = v_max
    #             v = v_max * np.cos(alpha)
    #         return v, w

    def pure_pursuit(self, d, phi, last_v, last_w, vehicle_msg):

        print(vehicle_msg)
        v_max = 0.25

        K = 4
        L = 0.5
        if d == None or phi == None or np.isnan(d) or np.isnan(phi) or d >= L:
            v = last_v
            w = last_w
            return v, w
        else:

            x = np.arcsin(d / L)
            alpha = -x - phi
            sin_alpha = np.sin(alpha)

            w = sin_alpha * K
            # v = v_max * np.cos(alpha)
            v = v_max
            return v, w