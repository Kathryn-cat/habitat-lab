import numpy as np
import magnum as mn
import habitat
import habitat_sim
from habitat.config import Config
from habitat.core.registry import registry
from habitat_sim.utils import viz_utils as vut
from kitchen.utils.visualize import simulate, visualize_image

def init_agent(sim, pos, ori):
    agent_pos = np.array(pos)

    # Place the agent
    sim.agents[0].scene_node.translation = agent_pos
    sim.agents[0].scene_node.rotation = mn.Quaternion.rotation(
        mn.Deg(ori), mn.Vector3(0, 1, 0) 
    )

