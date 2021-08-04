import habitat_sim
import new_configs.utils.paths as paths
import new_configs.utils.simulator as simulator
import new_configs.utils.visualization as visual

if __name__ == '__main__':
		cfg = simulator.make_cfg(simulator.settings)
		with habitat_sim.Simulator(cfg) as sim:
				simulator.init_agent(sim)
				# Visualize the agent's initial position
				visual.simulate_and_make_vid(
						sim, None, "apt_0", dt=1.0
				)

