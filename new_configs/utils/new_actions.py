import habitat_sim
import magnum as mn
import numpy as np

def raycast(sim, direction, origin, max_distance=0.2):
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

