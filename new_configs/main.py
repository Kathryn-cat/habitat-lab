import habitat_sim
from habitat_sim.utils.common import quat_from_magnum
import numpy as np
import magnum as mn
import new_configs.utils.paths as paths
import new_configs.utils.simulator as simulator
import new_configs.utils.visualization as visual

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

        # visualization
        visual.simulate_and_make_vid(
            sim, None, "apt_0_v1", dt=2.0
        )

