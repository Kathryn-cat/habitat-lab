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
from kitchen.utils.visualize import simulate, visualize_image, visualize_scene
from kitchen.utils.spawn_agent import init_agent
from kitchen.utils.configs import make_cfg

# define global variables 
dir_path = os.getcwd()
data_path = os.path.join(dir_path, "datasets")
output_path = os.path.join(dir_path, "kitchen/results")

if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument('--width', type=int, default=1280)
		parser.add_argument('--height', type=int, default=720)
		parser.add_argument('--gpu_id', type=str, default='0')
		parser.add_argument('--scene', type=str, default=\
												'datasets/ReplicaCAD/configs/scenes/apt_0.scene_instance.json')
		parser.add_argument('--scene_dataset', type=str, default=\
												'datasets/ReplicaCAD/replicaCAD.scene_dataset_config.json')
		parser.add_argument('--agent_pos', type=str, default='0 0 1')
		parser.add_argument('--agent_ori', type=float, default=90)
		parser.add_argument('--video_prefix', type=str, default='kitchen_env')
		args = parser.parse_args()

		# split the strings in args
		agent_pos = list(map(float, args.agent_pos.split(' ')))

		# set gpu 
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

		settings = {
				"gpu_id": int(args.gpu_id),
				"max_frames": 10,
				"width": args.width,
				"height": args.height,
				"scene": args.scene,
				"scene_dataset": args.scene_dataset,
				"default_agent_id": 0,
				"sensor_height": 1.5,  # Height of sensors in meters
				"sensor_pitch": 0,  # rotation in rads
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

		global sim
		global obj_attr_mgr
		global prim_attr_mgr
		global stage_attr_mgr
		global rigid_obj_mgr
		global metadata_mediator

		sim = None
		obj_attr_mgr = None
		prim_attr_mgr = None
		stage_attr_mgr = None
		rigid_obj_mgr = None
		metadata_mediator = None

		if sim != None:
				sim.close()

		## initialize simulator 
		cfg = make_cfg(settings)
		sim = habitat_sim.Simulator(cfg)

		# manage attribute templates 
		obj_attr_mgr = sim.get_object_template_manager()
		obj_attr_mgr.load_configs(str(os.path.join(data_path, "ReplicaCAD/configs/objects")))
		prim_attr_mgr = sim.get_asset_template_manager()
		stage_attr_mgr = sim.get_stage_template_manager()
		rigid_obj_mgr = sim.get_rigid_object_manager()
		metadata_mediator = sim.metadata_mediator

		# spawn the agent in the kitchen env
		init_agent(sim, agent_pos, args.agent_ori)
		
