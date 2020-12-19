#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows and toggles visual servoing to go park behind another duckie bot if detected
"""
import argparse
import logging
import sys

import gym
import numpy as np
import pyglet
from gym_duckietown.envs import DuckietownEnv
from PIL import Image
from pyglet.window import key

from config import (BUMPER_TO_CENTER_DIST, CAMERA_MODE, CIRCLE_MIN_AREA,
                    CIRCLE_MIN_DISTANCE, CIRCLE_PATTERN_HEIGHT,
                    CIRCLE_PATTERN_WIDTH, TARGET_DIST)
from control import Trajectory
from estimation import PoseEstimator


def main(duckie_env: DuckietownEnv, debug: bool):
    """
    Main loop that allows to control duckiebot with keyboard and uses visual servo when bot is detected
    Args:
        duckie_env: the environment in which our duckiebot evolves
        debug: will log debug message if True
    """
    duckie_env.reset()
    duckie_env.render()

    logger = logging.getLogger(__name__)
    if debug:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    pose_estimator = PoseEstimator(min_area=CIRCLE_MIN_AREA,
                                   min_dist_between_blobs=CIRCLE_MIN_DISTANCE,
                                   height=CIRCLE_PATTERN_HEIGHT,
                                   width=CIRCLE_PATTERN_WIDTH,
                                   target_distance=TARGET_DIST,
                                   camera_mode=CAMERA_MODE)

    # This is the object that computes the next command from the estimated pose
    trajectory = Trajectory()

    @duckie_env.unwrapped.window.event
    def on_key_press(symbol, modifier):
        """
        This handler processes keyboard commands that
        control the simulation

        Args:
            symbol: key pressed
        """
        if symbol in [key.BACKSPACE, key.SLASH]:
            logger.info("RESET")
            trajectory.reset()
            duckie_env.reset()
            duckie_env.render()
        elif symbol == key.PAGEUP:
            duckie_env.unwrapped.cam_angle[0] = 0
        elif symbol == key.ESCAPE:
            duckie_env.close()
            sys.exit(0)

    # Register a keyboard handler
    key_handler = key.KeyStateHandler()
    duckie_env.unwrapped.window.push_handlers(key_handler)

    def update(dt: float):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing

        Args:
            dt: change in time (in secs) since last update
        """
        action = np.array([0.0, 0.0])

        if key_handler[key.UP]:
            action += np.array([0.44, 0.0])
        if key_handler[key.DOWN]:
            action -= np.array([0.44, 0])
        if key_handler[key.LEFT]:
            action += np.array([0, 1])
        if key_handler[key.RIGHT]:
            action -= np.array([0, 1])
        # Speed boost
        if key_handler[key.LSHIFT]:
            action *= 1.5

        # TODO get observation before taking step
        # For now, we do nothing, get image, compute action, execute it. So 2 steps for one action
        # It should be get observations, compute action, take step, repeat. (1 action/step)

        obs, reward, done, info = duckie_env.step(action)

        target_detected, estimated_pose = pose_estimator.get_pose(obs)
        print(target_detected)

        # Only for debugging, slows things down considerably and is not necessary
        # if detect:
        #     cv2.drawChessboardCorners(obs,
        #                               (8, 3), centers, detection)
        #     im = Image.fromarray(obs)
        #     im.save("circle_grid.png")

        # Here we get the ground_truth to see the accuracy of our estimate
        # Note: This is in global frame, while the estimate is in robot frame.
        # Also, the exact distance from the center of the duck to the bumper is unknown so it is set approximately
        goal_position = np.array([2.5 - TARGET_DIST - BUMPER_TO_CENTER_DIST,
                                  0,
                                  2.5])  # Because duckie is at [2.5, 0. 2.5] in visual_servo.yaml env file
        goal_angle = 0  # Because duckie faces 0 angle in visual_servo.yaml env file
        cur_position = np.array(info["Simulator"]["cur_pos"])
        cur_angle_rad = info["Simulator"]["cur_angle"]
        cur_angle_deg = np.rad2deg(cur_angle_rad)
        if cur_angle_deg > 179:
            cur_angle_deg -= 360
        relative_position = goal_position - cur_position
        relative_angle = goal_angle - cur_angle_deg
        relative_pose = [relative_position, relative_angle]
        np.set_printoptions(precision=2)
        logger.debug(f"gt: {relative_pose}, estimate: {estimated_pose}")

        if target_detected:
            trajectory.update(estimated_pose)
        # elif trajectory.is_initialized():
            # trajectory.predict(left_encoder_delta=1, right_encoder_delta=1) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            logger.warning("object not found, cannot compute initial trajectory")
            # TODO for now we can move the duckie with the arrows. Eventually we just want
            # to reset the environment, and maybe log the starting pose to plot where we detect or not.

        if trajectory.is_initialized():
            action = trajectory.get_commands()
            obs, reward, done, info = duckie_env.step(action)

        if key_handler[key.RETURN]:
            im = Image.fromarray(obs)
            im.save("screen.png")

        if done:
            logger.info("done!")
            duckie_env.reset()
            duckie_env.render()

        duckie_env.render()

    pyglet.clock.schedule_interval(update, 1.0 / duckie_env.unwrapped.frame_rate)

    # Enter main event loop
    pyglet.app.run()

    duckie_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Duckietown-udem1-v0")
    parser.add_argument("--map-name", default="visual_servo")
    parser.add_argument("--distortion", default=False, action="store_true")
    parser.add_argument("--camera_rand", default=False, action="store_true")
    parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
    parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
    parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
    parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
    parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    if args.env_name and args.env_name.find("Duckietown") != -1:
        env = DuckietownEnv(
            seed=args.seed,
            map_name=args.map_name,
            draw_curve=args.draw_curve,
            draw_bbox=args.draw_bbox,
            domain_rand=args.domain_rand,
            frame_skip=args.frame_skip,
            distortion=args.distortion,
            camera_rand=args.camera_rand,
            dynamics_rand=args.dynamics_rand,
            accept_start_angle_deg=20,
            full_transparency=True
        )
    else:
        env = gym.make(args.env_name)
    main(env, args.debug)
