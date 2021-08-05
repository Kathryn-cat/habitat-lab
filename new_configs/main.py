import habitat_sim
from habitat_sim.utils.common import quat_from_magnum
import numpy as np
import magnum as mn
import new_configs.utils.paths as paths
import new_configs.utils.simulator as simulator
import new_configs.utils.visualization as visual
import new_configs.utils.new_actions as actions
import new_configs.utils.objects as objects

if __name__ == '__main__':
    cfg = simulator.make_cfg(simulator.settings)
    with habitat_sim.Simulator(cfg) as sim:

        # initialize agent
        initial_state = sim.agents[0].state
        initial_state.position = np.array([0.0, 0.0, 1.0])
        initial_state.rotation = quat_from_magnum(
            mn.Quaternion.rotation(mn.Deg(90), mn.Vector3(0.0, 1.0, 0.0))
        )
        sim.initialize_agent(agent_id=0, initial_state=initial_state)

        '''
        # visualization
        visual.simulate_and_make_vid(
            sim, None, "apt_0_v1", dt=2.0
        )

        # print all objects info 
        objects.print_object_info(sim)

        # test ray casting
        # ensure that cast_direction is a unit vector, because max_distance is in units of ray length
        cast_direction = mn.Vector3([0.0, -0.1, 0.0]).normalized()
        cast_origin = mn.Vector3([-2.0298, 1.0642, 2.6329])
        object_id, dist, point = actions.raycast(
            sim, cast_direction, cast_origin, max_distance=0.12
        )
        print(f"closest object id: {object_id}, distance: {dist}, point: {point}")
        '''

        # perform a series of robot motions and make a video
        imgs = []
        for i in range(30):
            obs = sim.step('move_forward')
            img = visual.agent_motion_img(sim, obs)
            imgs.append(img)
        visual.skvideo_from_imgs(imgs, 'agent_motion_v1.mp4')

