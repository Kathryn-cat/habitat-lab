import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import skvideo.io
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import quaternion
import os

ROBOT_HEAD_DIR = 'images/pick_task/robot_head'
ARM_DIR = 'images/pick_task/arm'
THIRD_DIR = 'images/pick_task/3rd'
FIXED_CAMERA_DIR = 'images/pick_task/fixed_camera'
FIXED_CAMERA_VDIR = 'videos/pick_task/fixed_camera'

deg2rad = lambda i : i / 180 * 3.14159
FIXED_CAMERA_EULER_1 = np.array([deg2rad(0), deg2rad(0), deg2rad(0)])
FIXED_CAMERA_POS_1 = np.array([0.8, -0.2, 0.7])

def save_rgb_images(obs, file_name):
		robot_head_rgb = obs['robot_head_rgb'][:, :, ::-1]
		arm_rgb = obs['arm_rgb'][:, :, ::-1]
		third_rgb = obs['3rd_rgb'][:, :, ::-1]
		robot_head_filepath = os.path.join(ROBOT_HEAD_DIR, file_name)
		arm_filepath = os.path.join(ARM_DIR, file_name)
		third_filepath = os.path.join(THIRD_DIR, file_name)
		cv2.imwrite(robot_head_filepath, robot_head_rgb)
		cv2.imwrite(arm_filepath, arm_rgb)
		cv2.imwrite(third_filepath, third_rgb)

def fixed_camera_images(env, position, euler):
		initial_state = env._sim.get_agent_state(0)
		init_position = initial_state.position
		position = init_position + position
		rotation = quaternion.from_euler_angles(euler)
		obs_args = dict(position=position, rotation=rotation, keep_agent_at_new_pose=True)
		obs = env._sim.get_observations_at(**obs_args)
		fixed_camera_img = obs['robot_head_rgb']
		fixed_camera_img = cv2.rotate(fixed_camera_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
		return fixed_camera_img
		
# randomly permute the rotation
def uniform_quat(original_angle):
    original_euler = quaternion.as_euler_angles(original_angle)
    euler_angles = np.array([np.random.rand() * np.pi * 2 + original_euler[0],
                             np.random.rand() * np.pi + original_euler[1],
                             np.random.rand() * np.pi * 2 + original_euler[2]])
    quaternions = quaternion.from_euler_angles(euler_angles)
    return quaternions

# generate random views 
def generate_views(env, num_views, exp_name):
		initial_state = env._sim.get_agent_state(0)
		init_rotation = initial_state.rotation
		init_position = initial_state.position
		file_name = exp_name + '_camera_angle_'
		for i in range(num_views):
				rotation = uniform_quat(init_rotation)
				position = init_position + np.random.rand(3,) * 3
				obs_args = dict(position=position, rotation=rotation, keep_agent_at_new_pose=True)
				obs = env._sim.get_observations_at(**obs_args)
				save_rgb_images(obs, file_name + str(i) + '.png')
				print(f'Round {i}: Position {position}; Rotation {rotation}')
				print(f'----------------------')

def make_video(stack_imgs, file_name):
		video = np.stack(stack_imgs, axis=0)
		video_filepath = os.path.join(FIXED_CAMERA_VDIR, file_name)
		video_args = dict(inputdict={'-r': str(5)}, outputdict={'-f': 'mp4', '-pix_fmt': 'yuv420p'})
		skvideo.io.vwrite(video_filepath, video, **video_args)

def pick_task(exp_name):
		env_args = dict(config=habitat.get_config('configs/tasks/rearrangepick_replica_cad_1.yaml'))
		env = habitat.Env(**env_args)
		obs = env.reset()
		def random_agent():
				num_steps = 0
				stack_imgs = []
				while not env.episode_over:
						action = env.action_space.sample()
						obs = env.step(action)
						print(f"obj_start_sensor: {obs['obj_start_sensor']}")
						print(f"obj_goal_sensor: {obs['obj_goal_sensor']}")
						print(f"action: {action}")
						print(f"-------------------------------------")
						stack_imgs.append(fixed_camera_images(env, FIXED_CAMERA_POS_1, FIXED_CAMERA_EULER_1))
						num_steps += 1
				file_name = exp_name + '.mp4'
				make_video(stack_imgs, file_name)
				print(f'Pick task finishes in {num_steps} steps')
		random_agent()

if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument('--exp_name', type=str, default='exp_0')
		args = parser.parse_args()

		np.random.seed(2)
		
		exp_args = dict(exp_name=args.exp_name)
		pick_task(**exp_args)

