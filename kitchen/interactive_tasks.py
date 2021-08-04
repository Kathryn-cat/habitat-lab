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

def get_rotation(sim, object_id):
    quat = sim.get_rotation(object_id)
    return np.array(quat.vector).tolist() + [quat.scalar]

def init_episode_dict(episode_id, scene_id, agent_pos, agent_rot):
    episode_dict = {
        "episode_id": episode_id,
        "scene_id": "data/scene_datasets/coda/coda.glb",
        "start_position": agent_pos,
        "start_rotation": agent_rot,
        "info": {},
    }
    return episode_dict

def add_object_details(sim, episode_dict, obj_id, object_template, object_id):
    object_template = {
        "object_id": obj_id,
        "object_template": object_template,
        "position": np.array(sim.get_translation(object_id)).tolist(),
        "rotation": get_rotation(sim, object_id),
    }
    episode_dict["objects"] = object_template
    return episode_dict

def add_goal_details(sim, episode_dict, object_id):
    goal_template = {
        "position": np.array(sim.get_translation(object_id)).tolist(),
        "rotation": get_rotation(sim, object_id),
    }
    episode_dict["goals"] = goal_template
    return episode_dict

# set the number of objects to 1 always for now.
def build_episode(sim, episode_num, object_id, goal_id):
    episodes = {"episodes": []}
    for episode in range(episode_num):
        agent_state = sim.get_agent(0).get_state()
        agent_pos = np.array(agent_state.position).tolist()
        agent_quat = agent_state.rotation
        agent_rot = np.array(agent_quat.vec).tolist() + [agent_quat.real]
        episode_dict = init_episode_dict(
            episode, settings["scene"], agent_pos, agent_rot
        )

        object_attr = sim.get_object_initialization_template(object_id)
        object_path = os.path.relpath(
            os.path.splitext(object_attr.render_asset_handle)[0]
        )

        episode_dict = add_object_details(
            sim, episode_dict, 0, object_path, object_id
        )
        episode_dict = add_goal_details(sim, episode_dict, goal_id)
        episodes["episodes"].append(episode_dict)

    return episodes
