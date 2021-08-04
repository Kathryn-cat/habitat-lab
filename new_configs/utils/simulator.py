import new_configs.utils.visualization
import habitat_sim
import numpy as np
import magnum as mn

# render RGB, depth 256 * 256 observations
# action space: move_forward, turn_left, turn_right 

def make_cfg(settings):
    # create simulator
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.default_agent_id = settings["default_agent_id"]
    sim_cfg.enable_physics = settings["enable_physics"]
    sim_cfg.physics_config_file = settings["physics_config_file"]
    sim_cfg.random_seed = settings["seed"]
    sim_cfg.scene_id = settings["scene_id"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.scene_light_setup = settings["scene_light_setup"]

    # create rgb and depth sensors
    # Note: all sensors must have the same resolution
    sensor_specs = []

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.hfov = settings["hfov"]
    rgb_sensor_spec.position = np.array([0.0, settings["sensor_height"], 0.0])
    rgb_sensor_spec.orientation = settings["orientation"]
    rgb_sensor_spec.resolution = np.array([settings["height"], settings["width"]])
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgb_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.hfov = settings["hfov"]
    depth_sensor_spec.position = np.array([0.0, settings["sensor_height"], 0.0])
    depth_sensor_spec.orientation = settings["orientation"]
    depth_sensor_spec.resolution = np.array([settings["height"], settings["width"]])
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    # create agent 
    # TODO: use new actions
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1, constraint=None)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0, constraint=None)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0, constraint=None)
        ),
    }
    agent_cfg.height = settings["agent_height"]
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# pre-defined configs
settings = {
    "default_agent_id": 0,
    "enable_physics": True,
    "physics_config_file": "data/default.physics_config.json",
    "seed": 1,
    "scene_id": "datasets/ReplicaCAD/configs/scenes/apt_0.scene_instance.json",
    "scene_dataset": "datasets/ReplicaCAD/replicaCAD.scene_dataset_config.json",
    "scene_light_setup": "datasets/ReplicaCAD/configs/lighting/frl_apartment_0.lighting_config.json",
    "hfov": mn.Deg(90),
    "orientation": np.array([0.0, 0.0, 0.0]),
    "sensor_height": 1.5,
    "width": 256,
    "height": 256,
    "agent_height": 1.5,
}

# Spawn the agent at a pre-defined location
def init_agent(sim):
    agent_pos = mn.Vector3(0.0, 0.0, 1.0)
    agent_rot = mn.Deg(90)

    # Place the agent
    sim.agents[0].scene_node.translation = agent_pos
    sim.agents[0].scene_node.rotation = mn.Quaternion.rotation(
        agent_rot, mn.Vector3(0.0, 1.0, 0.0)
    )

