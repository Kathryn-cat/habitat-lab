import habitat_sim
import magnum as mn
import numpy as np
import attr
from typing import Any, Dict, List, Optional, Type
from habitat.core.registry import registry
from habitat.config import Config

# direction: List[float]
# origin: List[float]
def raycast(sim, direction, origin, max_distance=0.2):
    direction = mn.Vector3(direction).normalized()
    origin = mn.Vector3(origin)
    gripper_ray = habitat_sim.geo.Ray(origin, direction)
    raycast_results = sim.cast_ray(gripper_ray, max_distance=max_distance)
    ids = sim.get_existing_object_ids()

    closest_object_id = -1
    closest_dist = 1000.0
    closest_object_point = None
    
    if raycast_results.has_hits():
        print(f'Total number of hits: {len(raycast_results.hits)}')
        for hit in raycast_results.hits:
            if hit.ray_distance < closest_dist and hit.object_id in ids:
                closest_dist = hit.ray_distance
                closest_object_id = hit.object_id
                closest_object_point = hit.point
    else:
        print(f'Total number of hits: 0')

    return closest_object_id, closest_dist, closest_object_point

# define a grab / release action
# TODO: delete this line and modify the default code 

from habitat.config.default import _C, CN
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat_sim.agent.controls.controls import ActuationSpec
from habitat_sim.physics import MotionType

@attr.s(auto_attribs=True, slots=True)
class GrabReleaseActuationSpec(ActuationSpec):
    direction: List[float] = [0.0, -0.1, 0.0]
    origin: List[float] = [-2.0298, 1.0642, 2.6329]
    max_distance: float = 0.2

@registry.register_action_space_configuration(name="RearrangementActions-v0")
class RearrangementSimV0ActionSpaceConfiguration(
    HabitatSimV1ActionSpaceConfiguration
):
    def __init__(self, config):
        super().__init__(config)
        if not HabitatSimActions.has_action("GRAB_RELEASE"):
            HabitatSimActions.extend_action_space("GRAB_RELEASE")

    def get(self):
        config = super().get()
        new_config = {
            HabitatSimActions.GRAB_RELEASE: habitat_sim.ActionSpec(
                "grab_or_release_object_under_crosshair",
                GrabReleaseActuationSpec(
                    direction=self.config.RAYCAST_DIRECTION,
                    origin=self.config.RAYCAST_ORIGIN,
                    max_distance=self.config.MAX_DISTANCE,
                ),
            )
        }

        config.update(new_config)

        return config

@registry.register_task_action
class GrabOrReleaseAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        return self._sim.step(HabitatSimActions.GRAB_RELEASE)

