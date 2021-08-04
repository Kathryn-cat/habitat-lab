import habitat_sim

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
    for i in range(num_objects):
        print(f'\
            index: {i}\t\
            object id: {ids[i]}\t\
            handle: {handles[i]}\t\
            motion type: {motion_types[i]}\t\
            collidable: {collidable[i]}\t\
            local collision mesh: {local_bounding_box[i]}\t\
            global com: {com[i]}\t\
            translation wrt SceneNode: {scenenode_translation[i]}\t\
            rotation wrt SceneNode: {scenenode_rotation[i]}\
        ')

