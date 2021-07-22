import numpy as np
import magnum as mn
import habitat
import habitat_sim
from habitat.config import Config
from habitat.core.registry import registry
from habitat_sim.utils import viz_utils as vut
from kitchen.utils.visualize import make_video_cv2, simulate, simulate_and_make_vid, \
																		save_display_sample

def init_agent(sim, pos):
    agent_pos = np.array(pos)

    # Place the agent
    sim.agents[0].scene_node.translation = agent_pos
    agent_orientation_y = -40 # can change 
    sim.agents[0].scene_node.rotation = mn.Quaternion.rotation(
        mn.Deg(agent_orientation_y), mn.Vector3(0, 1.0, 0) # can change
    )

