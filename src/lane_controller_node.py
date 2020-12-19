#!/usr/bin/env python3
import numpy as np
import rospy
import os
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, \
    WheelEncoderStamped

from lane_controller.controller import PurePursuitLaneController
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from visual_servo.estimation import PoseEstimator
from visual_servo.control import Trajectory
from visual_servo.config import (BUMPER_TO_CENTER_DIST, CAMERA_MODE, CIRCLE_MIN_AREA,
                                 CIRCLE_MIN_DISTANCE, CIRCLE_PATTERN_HEIGHT,
                                 CIRCLE_PATTERN_WIDTH, TARGET_DIST)


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.pp_controller = PurePursuitLaneController(self.params)

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers

        self.sub_image = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/camera_node/image/compressed",
                                          CompressedImage, self.cb_image, queue_size=1)

        self.sub_info = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/camera_node/camera_info", CameraInfo,
                                         self.cb_process_camera_info, queue_size=1)

        self.sub_encoder_left = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/left_wheel_encoder_node/tick",
                                                 WheelEncoderStamped,
                                                 self.cbProcessLeftEncoder,
                                                 queue_size=1)

        self.sub_encoder_right = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/right_wheel_encoder_node/tick",
                                                  WheelEncoderStamped,
                                                  self.cbProcessRightEncoder,
                                                  queue_size=1)
        self.log("Initialized!")


        self.right_encoder_ticks = 0
        self.left_encoder_ticks = 0
        self.right_encoder_ticks_delta = 0
        self.left_encoder_ticks_delta = 0

        # Set up a timer for prediction (if we got encoder data) since that data can come very quickly
        self._predict_freq = rospy.get_param('~predict_frequency', 30.0)

        rospy.Timer(rospy.Duration(1 / self._predict_freq), self.cbPredict)

        self.bridge = CvBridge()

        self.last_stamp = rospy.Time.now()

        self.pcm = PinholeCameraModel()
        self.pose_estimator = PoseEstimator(min_area=CIRCLE_MIN_AREA,
                                            min_dist_between_blobs=CIRCLE_MIN_DISTANCE,
                                            height=CIRCLE_PATTERN_HEIGHT,
                                            width=CIRCLE_PATTERN_WIDTH,
                                            target_distance=TARGET_DIST,
                                            camera_mode=CAMERA_MODE,
                                            )
        self.trajectory = Trajectory()

    def cbProcessLeftEncoder(self, left_encoder_msg):
        """
        Calculate the number of ticks since the last time
        """

        self.left_encoder_ticks_delta = left_encoder_msg.data - self.left_encoder_ticks

    def cbProcessRightEncoder(self, right_encoder_msg):
        """
        Calculate the number of ticks since the last time
        """

        self.right_encoder_ticks_delta = right_encoder_msg.data - self.right_encoder_ticks

    def cbPredict(self, event):
        """
        Predict the pose of the duckiebot since the last detection with the encoder data.
        """



        # first let's check if we moved at all, if not abort
        if self.right_encoder_ticks_delta == 0 and self.left_encoder_ticks_delta == 0:
            return

        self.left_encoder_ticks += self.left_encoder_ticks_delta
        self.right_encoder_ticks += self.right_encoder_ticks_delta

        self.trajectory.predict(self.left_encoder_ticks_delta, self.right_encoder_ticks_delta)

        self.left_encoder_ticks_delta = 0
        self.right_encoder_ticks_delta = 0

    def cb_process_camera_info(self, msg):
        """
        Callback that stores the intrinsic calibration into a PinholeCameraModel object.

        Args:

            msg (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties of the camera.
        """
        if not self.pose_estimator.initialized:
            self.pcm.fromCameraInfo(msg)
            self.pose_estimator.initialize_camera_matrix(self.pcm.intrinsicMatrix(), self.pcm.distortionCoeffs())

    def cb_image(self, image_msg):
        """
        From the image, the function estimate the pose of the duckiebot and send the
        commands to follow the trajectory.
        """

        now = rospy.Time.now()

        dt = (now - self.last_stamp).to_sec()

        self.last_stamp = now

        if self.pose_estimator.initialized:
            image_cv = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")

            target_detected, estimated_pose = self.pose_estimator.get_pose(image_cv)



            if target_detected:
                self.trajectory.update(estimated_pose)
                self.left_encoder_ticks_delta = 0
                self.right_encoder_ticks_delta = 0

            v = 0
            w = 0

            if self.trajectory.is_initialized():
                v, w = self.trajectory.get_commands()
            car_control_msg = Twist2DStamped()
            car_control_msg.header = image_msg.header
            car_control_msg.v = v
            car_control_msg.omega = w

            self.publishCmd(car_control_msg)

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)

    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
