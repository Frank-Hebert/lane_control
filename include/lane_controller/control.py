"""
This module is computing duckiebot commands from estimated pose to a target
"""
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class Trajectory:
    """
    Class that handles the trajectory logic
    """

    def __init__(self,
                 distance_threshold,
                 final_angle_threshold,
                 distance_wheel,
                 wheel_radius,
                 target_dist
                 ):
        self.distance_threshold = distance_threshold
        self.final_angle_threshold = final_angle_threshold
        self.distance_wheel = distance_wheel
        self.wheel_radius = wheel_radius
        self.target_dist = target_dist

        self.distance_to_target = 0.0
        self.angle_to_target = 0.0
        self.angle_to_goal_pose = 0.0
        self.last_update_time = None
        self.last_commands = None
        self.done = False
        self.target_in_sight = False
        self.initialized = False
        self.relative_pose = None

    def is_initialized(self) -> bool:
        """
        getter method to know if trajectory was initalized
        Returns:
            True if we have seen the object at least once to compute initial trajectory
        """
        return self.initialized

    def reset(self):
        """
        Resets trajectory
        """
        self.distance_to_target = 0.0
        self.angle_to_target = 0.0
        self.angle_to_goal_pose = 0.0
        self.last_update_time = None
        self.last_commands = None
        self.done = False
        self.target_in_sight = False
        self.initialized = False

    def update(self, relative_pose: Tuple[np.array, float]):
        """
        Update our belief on target location from estimated pose
        Args:
            relative_pose: A tuple ([y, z, x], theta) giving pose relative to target
        """
        if not self.done:
            x_dist = relative_pose[0][2]
            y_dist = relative_pose[0][0]
            self.relative_pose = relative_pose
            self.distance_to_target = np.sqrt(x_dist ** 2 + y_dist ** 2)
            self.angle_to_target = np.rad2deg(np.arctan(y_dist / x_dist))
            self.angle_to_goal_pose = relative_pose[1]
            self.target_in_sight = True
            self.initialized = True

    def get_commands(self) -> np.array:
        """
        Get next duckiebot commands from current belief
        Returns:
            a np.array [v, w]
        """

        alpha = - np.arctan(self.relative_pose[0][0] / self.relative_pose[0][2])
        v = 0.7 * np.abs(self.relative_pose[0][2] - self.target_dist) / 0.7
        w = 2 * np.sin(alpha)

        # Stop the duckiebot if it's in the range of the target
        if abs(np.rad2deg(alpha)) < self.final_angle_threshold:
            w = 0
        if np.abs(self.relative_pose[0][2] - self.target_dist) < self.distance_threshold:
            v = 0

        commands = np.array([v, w])
        self.last_commands = commands
        return commands

    def predict(self, left_encoder_delta: int, right_encoder_delta: int):
        """
        update belief according to kinematics when no object is detected
        Args:
            left_encoder_delta:
            right_encoder_delta:
        """
        if not self.done:
            d_r = (right_encoder_delta * 2 * np.pi / 135 * 2 * self.wheel_radius)  # Movement right wheel
            d_l = (left_encoder_delta * 2 * np.pi / 135 * 2 * self.wheel_radius)  # Movement left wheel

            d = 0.5 * (d_r + d_l)
            d_omega = 0.5 * (d_r - d_l) / self.distance_wheel / 2

            self.distance_to_target -= d
            self.angle_to_target -= np.rad2deg(d_omega)
            self.angle_to_goal_pose -= np.rad2deg(d_omega)
            self.target_in_sight = False
