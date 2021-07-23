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

def make_video_cv2(
    observations, cross_hair=None, prefix="", open_vid=False, fps=60
):
    sensor_keys = list(observations[0])
    videodims = observations[0][sensor_keys[0]].shape
    videodims = (videodims[1], videodims[0])  # flip to w,h order
    print(videodims)
    video_file = os.path.join(output_path, prefix + ".mp4")
    print("Encoding the video: %s " % video_file)
    writer = vut.get_fast_video_writer(video_file, fps=fps)
    for ob in observations:
        # If in RGB/RGBA format, remove the alpha channel
        rgb_im_1st_person = cv2.cvtColor(ob["rgb"], cv2.COLOR_RGBA2RGB)
        if cross_hair is not None:
            rgb_im_1st_person[
                cross_hair[0] - 2 : cross_hair[0] + 2,
                cross_hair[1] - 2 : cross_hair[1] + 2,
            ] = [255, 0, 0]

        if rgb_im_1st_person.shape[:2] != videodims:
            rgb_im_1st_person = cv2.resize(
                rgb_im_1st_person, videodims, interpolation=cv2.INTER_AREA
            )
        # write the 1st person observation to video
        writer.append_data(rgb_im_1st_person)
    writer.close()

    if open_vid:
        print("Displaying video")
        vut.display_video(video_file)

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

def simulate_and_make_vid(sim, crosshair, prefix, dt=5.0, open_vid=False):
    observations = simulate(sim, dt)
    make_video_cv2(observations, crosshair, prefix=prefix, open_vid=open_vid)

def save_display_sample(
    rgb_obs,
    semantic_obs=np.array([]),
    depth_obs=np.array([]),
    key_points=None,  # noqa: B006
		prefix="sample",
):

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        # plot points on images
        if key_points is not None:
            for point in key_points:
                plt.plot(
                    point[0], point[1], marker="o", markersize=10, alpha=0.8
                )

    plt.savefig(os.path.join(output_path, prefix + '.png'))

