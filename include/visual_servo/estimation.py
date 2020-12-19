"""
This module contains the logic to estimate the pose of the duckie relative to a circle pattern from an image
"""
from typing import Optional, Tuple

import cv2
import numpy as np


class PoseEstimator:
    """
    Object that handle the pose estimation logic
    """

    def __init__(self,
                 min_area: int,
                 min_dist_between_blobs: int,
                 height: int,
                 width: int,
                 target_distance: float,
                 camera_mode: int):

        """
        Initialize PoseEstimator
        Args:
            min_area: minimum area of blob to be detected
            min_dist_between_blobs: minimum space between blob
            height: number of rows in circle pattern
            width: number of columns in circle pattern
            target_distance: target goal to bumper
            camera_mode: int from 0 to 3, temp solution for debugging
        """
        params = cv2.SimpleBlobDetector_Params()
        self.detector = cv2.SimpleBlobDetector_create(params)
        self.height = height
        self.width = width
        self.circle_pattern = self._calc_circle_pattern()
        self.target_distance = target_distance
        self.camera_matrix = None
        self.distortion_coefs = None
        self.camera_mode = camera_mode
        self.initialized = False

    def initialize_camera_matrix(self, camera_matrix, distortion_coefs):
        """
        Initialize the camera parameter
        """
        self.distortion_coefs = distortion_coefs
        self.camera_matrix = camera_matrix
        self.initialized = True

    def get_pose(self, obs: np.array) -> Tuple[bool, Optional[Tuple[np.array, float]]]:
        """
        Estimate a pose relative to a circle pattern, given an image
        Args:
            obs: input image in form of np.array

        Returns:
            a bool (True if pattern is detected), and a tuple([y, z, x], theta)
        """

        #Detect the circles pattern
        detection, centers = cv2.findCirclesGrid(obs,
                                                 patternSize=(self.width, self.height),
                                                 flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                                                 blobDetector=self.detector)

        #Estimate the pose from the circle pattern position
        if detection:
            image_points = centers[:, 0, :]
            _, rotation_vector, translation_vector = cv2.solvePnP(objectPoints=self.circle_pattern,
                                                                  imagePoints=image_points,
                                                                  cameraMatrix=self.camera_matrix,
                                                                  distCoeffs=self.distortion_coefs)
        else:
            return detection, None

        theta = rotation_vector[1][0]

        x_global = translation_vector[2][0]
        y_global = translation_vector[0][0]
        z_global = translation_vector[1][0]

        #We want to park behind the duckiebot at a distance target_distance

        x_but_global = x_global - self.target_distance
        y_but_global = y_global
        z_but_global = z_global

        x_but_robot = x_but_global * np.cos(theta)
        y_but_robot = y_but_global * np.cos(theta)
        z_but_robot = z_but_global

        return detection, (np.array([y_but_robot, z_but_robot, x_but_robot]), -np.rad2deg(theta))

    def _calc_circle_pattern(self):
        """
        Calculates the physical locations of each dot in the pattern.
        """
        circle_pattern_dist = 0.0125
        circle_pattern = np.zeros([self.height * self.width, 3])
        for i in range(self.width):
            for j in range(self.height):
                circle_pattern[i + j * self.width, 0] = circle_pattern_dist * i - \
                                                        circle_pattern_dist * (self.width - 1) / 2
                circle_pattern[i + j * self.width, 1] = circle_pattern_dist * j - \
                                                        circle_pattern_dist * (self.height - 1) / 2
        return circle_pattern
