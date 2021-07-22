# adapted from interactive tasks 
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import quaternion
import sys
import os
import gzip
import json
from typing import Any, Dict, List, Optional, Type
import attr
import magnum as mn

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import skvideo.io

import habitat
import habitat_sim
from habitat.config import Config
from habitat.core.registry import registry
from habitat_sim.utils import viz_utils as vut
from kitchen.utils.visualize import make_video_cv2, simulate, simulate_and_make_vid, \
																		save_display_sample
from kitchen.utils.spawn_agent import init_agent

## set up the simulator 
dir_path = os.getcwd()
data_path = os.path.join(dir_path, "datasets")
output_path = os.path.join(dir_path, "kitchen/results")

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = settings["gpu_id"]
    sim_cfg.default_agent_id = settings["default_agent_id"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    sim_cfg.physics_config_file = settings["physics_config_file"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgb_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument('--render_size', type=int, default=512)
		parser.add_argument('--gpu_id', type=str, default='0')
		parser.add_argument('--scene_path', type=str, default=\
										    'datasets/scene_datasets/habitat-test-scenes/apartment-1.glb')
		parser.add_argument('--init_agent_pos', type=str, default='0 0 0')
		args = parser.parse_args()

		# split the strings in args
		init_agent_pos = list(map(int, args.init_agent_pos.split(' ')))

		# set gpu 
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

		settings = {
				"gpu_id": int(args.gpu_id),
				"max_frames": 10,
				"width": args.render_size,
				"height": args.render_size,
				"scene": args.scene_path,
				"default_agent_id": 0,
				"sensor_height": 1.5,  # Height of sensors in meters
				"rgb": True,  # RGB sensor
				"depth": True,  # Depth sensor
				"seed": 1,
				"enable_physics": True,
				"physics_config_file": "datasets/default.physics_config.json",
				"silent": False,
				"compute_shortest_path": False,
				"compute_action_shortest_path": False,
				"save_png": True,
		}

		# set simulator configs 
		cfg = make_cfg(settings)

		## spawn the agent at pre-defined location 
		cfg.sim_cfg.default_agent_id = 0
		with habitat_sim.Simulator(cfg) as sim:
				init_agent(sim)
				# make video
				simulate_and_make_vid(
						sim, None, "init_position"
				)

