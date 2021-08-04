import habitat_sim
from tabulate import tabulate

def print_object_info(sim):
    mgr = sim.get_rigid_object_manager()
    num_objects = mgr.get_num_objects() 
    ids = sim.get_existing_object_ids()
    handles = []
    motion_types = []
    collidable = []
    local_bounding_box = []
    com = []
    scenenode_translation = []
    scenenode_rotation = []

    def MotionType2str(motion_type):
        if motion_type == habitat_sim.physics.MotionType.UNDEFINED:
            return "undefined"
        elif motion_type == habitat_sim.physics.MotionType.STATIC:
            return "static"
        elif motion_type == habitat_sim.physics.MotionType.KINEMATIC:
            return "kinematic"
        elif motion_type == habitat_sim.physics.MotionType.DYNAMIC:
            return "dynamic"
        return "error"
    
    for i in ids:
        handles.append(mgr.get_object_handle_by_id(i))
        obj = mgr.get_object_by_id(i)
        motion_types.append(MotionType2str(obj.motion_type))
        collidable.append(obj.collidable)
        local_bounding_box.append(obj.collision_shape_aabb)
        com.append(obj.com)
        scenenode_translation.append(obj.translation)
        scenenode_rotation.append(obj.rotation)

    print(f'NUMBER OF OBJECTS IN THE ENVIRONMENT: {num_objects}')
    data = []
    for i in range(num_objects):
        data_i = [\
            i, ids[i], handles[i], motion_types[i], collidable[i], local_bounding_box[i], \
            com[i], scenenode_translation[i], scenenode_rotation[i]\
        ]
        data.append(data_i)
    headers = ["index", "object id", "handle", "motion type", "collidable", \
               "local collision mesh", "global com", "translation w.r.t SceneNode", \
               "rotation w.r.t SceneNode"]
    print(tabulate(data, headers=headers))

