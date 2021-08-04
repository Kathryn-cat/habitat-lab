import habitat_sim
import magnum as mn
import numpy as np

def raycast(sim, direction, origin, max_distance=2.0):
    gripper_ray = habitat_sim.geo.Ray(direction, origin)
    raycast_results = sim.cast_ray(gripper_ray, max_distance=max_distance)

    closest_object = -1
    closest_dist = 1000.0
    if raycast_results.has_hits():
        for hit in raycast_results.hits:
            if hit.ray_distance < closest_dist:
                closest_dist = hit.ray_distance
                closest_object = hit.object_id

    return closest_object

