#!/usr/bin/env python3
import os

import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, DTParam, NodeType, ParamType, TopicType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, CompressedImage

from lane_controller.control import Trajectory
from lane_controller.estimation import PoseEstimator


class LaneControllerNode(DTROS):
    """
    Computes control action.
    The node compute the commands in form of linear and angular velocities.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        /['VEHICLE_NAME']/camera_node/image/compressed (:obj:`CompressedImage`): The camera image
        /['VEHICLE_NAME']/camera_node/camera_info (:obj:`CameraInfo`): The camera intrinsics parameters
        /['VEHICLE_NAME']/left_wheel_encoder_node/tick (:obj:`WheelEncoderStamped`): The left wheel encoder ticks
        /['VEHICLE_NAME']/right_wheel_encoder_node/tick (:obj:`WheelEncoderStamped`): The right wheel encoder ticks
    """

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary

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
                                                 self.cb_process_left_encoder,
                                                 queue_size=1)

        self.sub_encoder_right = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/right_wheel_encoder_node/tick",
                                                  WheelEncoderStamped,
                                                  self.cb_process_right_encoder,
                                                  queue_size=1)
        self.right_encoder_ticks = 0
        self.left_encoder_ticks = 0
        self.right_encoder_ticks_delta = 0
        self.left_encoder_ticks_delta = 0

        # Set up a timer for prediction (if we got encoder data) since that data can come very quickly
        self._predict_freq = rospy.get_param('~predict_frequency', 30.0)

        rospy.Timer(rospy.Duration(1 / self._predict_freq), self.cb_predict)

        self.bridge = CvBridge()

        self.pcm = PinholeCameraModel()

        self.pose_estimator = PoseEstimator(height=rospy.get_param("~circle_pattern_height"),
                                            width=rospy.get_param("~circle_pattern_width"),
                                            target_distance=rospy.get_param("~target_dist"),
                                            )

        self.trajectory = Trajectory(distance_threshold= rospy.get_param("~distance_threshold"),
                                     final_angle_threshold = rospy.get_param("~final_angle_threshold"),
                                     distance_wheel = rospy.get_param("~distance_wheel"),
                                     wheel_radius = rospy.get_param("~wheel_radius"),
                                     target_dist = rospy.get_param("~target_dist"))

    def cb_process_left_encoder(self, left_encoder_msg):
        """
        Update the number of left wheel ticks since the last trajectory update

        Args:
            left_encoder_msg (:obj:`WheelEncoderStamped`): message sent from left wheel encoder
        """
        self.left_encoder_ticks_delta = left_encoder_msg.data - self.left_encoder_ticks

    def cb_process_right_encoder(self, right_encoder_msg):
        """
        Update the number of right wheel ticks since the last trajectory update

        Args:
            right_encoder_msg (:obj:`WheelEncoderStamped`): message sent from right wheel encoder
        """
        self.right_encoder_ticks_delta = right_encoder_msg.data - self.right_encoder_ticks

    def cb_predict(self, event):
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
        Store the intrinsic calibration into a PinholeCameraModel object.

        Args:
            msg (:obj:`sensor_msgs.msg.CameraInfo`): Intrinsic properties of the camera.
        """
        if not self.pose_estimator.initialized:
            self.pcm.fromCameraInfo(msg)
            self.pose_estimator.initialize_camera_matrix(self.pcm.intrinsicMatrix(), self.pcm.distortionCoeffs())

    def cb_image(self, image_msg):
        """
        Estimate the pose of the duckiebot and send the commands to follow the trajectory.

        Args:
            image_msg (:obj:`CompressedImage`): Camera message containing compressed image
        """
        if self.pose_estimator.initialized:
            image_cv = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")

            target_detected, estimated_pose = self.pose_estimator.get_pose(image_cv)

            # Update the pose if the target is detected
            if target_detected:
                self.trajectory.update(estimated_pose)
                self.left_encoder_ticks_delta = 0
                self.right_encoder_ticks_delta = 0

            v = 0
            w = 0

            # Get commands from current estimate if pattern has been seen at least once
            if self.trajectory.is_initialized():
                v, w = self.trajectory.get_commands()
            car_control_msg = Twist2DStamped()
            car_control_msg.header = image_msg.header
            car_control_msg.v = v
            car_control_msg.omega = w

            self.publish_cmd(car_control_msg)

    def publish_cmd(self, car_cmd_msg):
        """
        Publish a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
