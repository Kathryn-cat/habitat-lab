import habitat_sim
import magnum as mn
import numpy as np
import attr
from typing import Any, Dict, List, Optional, Type
from habitat.core.registry import registry
from habitat.config import Config

# TODO: delete this line and modify the default code 
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim.nav import NavMeshSettings
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum

@registry.register_simulator(name="RearrangementSim-v0")
class RearrangementSim(HabitatSim):
    def __init__(self, config: Config) -> None:
        self.did_reset = False
        super().__init__(config=config)
        self.grip_offset = np.eye(4)

        agent_id = self.habitat_config.DEFAULT_AGENT_ID
        agent_config = self._get_agent_config(agent_id)
        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = agent_config.RADIUS
        self.navmesh_settings.agent_height = agent_config.HEIGHT

    def reset(self):
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations()

        self._prev_sim_obs = sim_obs
        self.did_reset = True
        self.grip_offset = np.eye(4)
        return self._sensor_suite.get_observations(sim_obs)

    def _sync_gripped_object(self, gripped_object_id):
        r"""
        Sync the gripped object with the object associated with the agent.
        """
        if gripped_object_id != -1:
            agent_body_transformation = (
                self._default_agent.scene_node.transformation
            )
            self.set_transformation(
                agent_body_transformation, gripped_object_id
            )
            translation = agent_body_transformation.transform_point(
                np.array([0, 2.0, 0])
            )
            self.set_translation(translation, gripped_object_id)

    @property
    def gripped_object_id(self):
        return self._prev_sim_obs.get("gripped_object_id", -1)

    def step(self, action: int):
        dt = 1 / 60.0
        self._num_total_frames += 1
        collided = False
        gripped_object_id = self.gripped_object_id

        agent_config = self._default_agent.agent_config
        action_spec = agent_config.action_space[action]
        if action_spec.name == "grab_or_release_object_under_crosshair":
            # If already holding an agent
            if gripped_object_id != -1:
                agent_body_transformation = (
                    self._default_agent.scene_node.transformation
                )
                T = np.dot(agent_body_transformation, self.grip_offset)

                self.set_transformation(T, gripped_object_id)

                position = self.get_translation(gripped_object_id)

                if self.pathfinder.is_navigable(position):
                    self.set_object_motion_type(
                        MotionType.STATIC, gripped_object_id
                    )
                    gripped_object_id = -1
                    self.recompute_navmesh(
                        self.pathfinder, self.navmesh_settings, True
                    )
            # if not holding an object, then try to grab
            else:
                gripped_object_id = raycast(
                    self,
                    direction=action_spec.actuation.direction,
                    origin=action_spec.actuation.origin,
                    max_distance=action_spec.actuation.max_distance,
                )

                # found a grabbable object.
                if gripped_object_id != -1:
                    agent_body_transformation = (
                        self._default_agent.scene_node.transformation
                    )
                    self.grip_offset = np.dot(
                        np.array(agent_body_transformation.inverted()),
                        np.array(self.get_transformation(gripped_object_id)),
                    )
                    self.set_object_motion_type(
                        MotionType.KINEMATIC, gripped_object_id
                    )
                    self.recompute_navmesh(
                        self.pathfinder, self.navmesh_settings, True
                    )

        else:
            collided = self._default_agent.act(action)
            self._last_state = self._default_agent.get_state()

        # step physics by dt
        super().step_world(dt)
        # Sync the gripped object after the agent moves.
        self._sync_gripped_object(gripped_object_id)

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations()
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = gripped_object_id

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations

