#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat
import habitat.tasks.expl.utils as utils

def expl():
    config = habitat.get_config("configs/tasks/expl.yaml")
    with habitat.Env(config=config) as env:
        obs = env.reset()
        imgs = []
        count_steps = 0
        while not env.episode_over:
            '''
            from habitat_sim.utils.common import quat_from_magnum
            import cv2; import magnum as mn; import pdb; pdb.set_trace()
            '''
            action = env.action_space.sample()
            obs = env.step(action)
            img = utils.agent_motion_img(env, obs, action)
            imgs.append(img)
            count_steps += 1
            print(f"count steps: {count_steps}")
        filename = 'videos/test_2.mp4'
        utils.skvideo_from_imgs(imgs, filename)

if __name__ == "__main__":
    expl()

