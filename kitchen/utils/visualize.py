import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
from habitat_sim.utils.common import d3_40_colors_rgb

dir_path = os.getcwd()
output_path = os.path.join(dir_path, "kitchen/results")

def simulate(sim, dt=5.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations

def visualize_image(sim, prefix):
		# simulate for a very short time 
		sim.step_physics(1.0 / 60.0)
		observation = sim.get_sensor_observations()['rgb'][:, :, :3]
		rgb_image = Image.fromarray(np.uint8(observation))
		rgb_image.save(os.path.join(output_path, prefix + '.png'))	

def visualize_scene(sim, prefix, dt=15.0, width=1280, height=720):
		observations = []
		start_time = sim.get_world_time()
		while sim.get_world_time() < start_time + dt:
				sim.agents[0].scene_node.rotate(mn.Rad(mn.math.pi * 2 / (60.0 * dt)), mn.Vector3(0, 1, 0))
				sim.step_physics(1.0 / 60.0)
				observations.append(sim.get_sensor_observations())

		# video rendering of carousel view
		vut.make_video(
				observations,
				"rgb",
				"color",
				os.path.join(output_path, prefix),
				open_vid=False,
				video_dims=[width, height],
		)

