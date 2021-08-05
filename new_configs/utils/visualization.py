import gzip
import json
import os
import sys
from typing import Any, Dict, List, Optional, Type
import attr
import cv2
import git
import skvideo.io
import magnum as mn
import numpy as np
import quaternion
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from PIL import Image
import habitat
import habitat_sim
from habitat.config import Config
from habitat.core.registry import registry
from habitat_sim.utils import viz_utils as vut
from new_configs.utils.paths import output_path

def make_video_cv2(
    observations, cross_hair=None, prefix="", open_vid=False, fps=60
):
    sensor_keys = list(observations[0])
    videodims = observations[0][sensor_keys[0]].shape
    videodims = (videodims[1], videodims[0])  # flip to w,h order
    print(f'videodims: {videodims}')
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

def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations

# convenience wrapper for simulate and make_video_cv2
def simulate_and_make_vid(sim, crosshair, prefix, dt=1.0, open_vid=False):
    observations = simulate(sim, dt)
    make_video_cv2(observations, crosshair, prefix=prefix, open_vid=open_vid)

def display_sample(
    rgb_obs,
    semantic_obs=np.array([]),
    depth_obs=np.array([]),
    key_points=None,  # noqa: B006
):
    print(f'WARNING: call plt.imshow()')

    from habitat_sim.utils.common import d3_40_colors_rgb

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
        plt.imshow(data)

    plt.show(block=False)

# print agent info on its first-person camera as well as third-party camera
def agent_motion_img(sim, obs):
    collided = obs['collided']
    rgb = obs['rgb'][:, :, :3][:, :, ::-1]
    agent_pos = np.round(sim.agents[0].state.position, 4)
    agent_rot = np.round(quaternion.as_float_array(sim.agents[0].state.rotation), 4)
    agent_vel = np.round(sim.agents[0].state.velocity, 4)
    agent_ang_vel = np.round(sim.agents[0].state.angular_velocity, 4)

    # annotate with text
    blank = np.zeros([850 - 512, 512, 3])

    cv2.putText(blank, f'collision: {collided}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, \
                0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(blank, f'position: {agent_pos}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, \
                0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(blank, f'rotation: {agent_rot}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, \
                0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(blank, f'velocity: {agent_vel}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, \
                0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(blank, f'angular velocity: {agent_ang_vel}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, \
                0.5, (255, 255, 0), 1, cv2.LINE_AA)

    img = np.concatenate([rgb, blank])
    return img

def skvideo_from_imgs(imgs, filename):
    video = np.stack(imgs, axis=0)[:, :, :, ::-1]
    filepath = os.path.join(output_path, filename)
    skvideo_args = dict(inputdict={'-r': str(5)}, outputdict={'-f': 'mp4', '-pix_fmt': 'yuv420p'})
    skvideo.io.vwrite(filepath, video, **skvideo_args)

