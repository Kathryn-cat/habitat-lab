#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat

def expl():
    config = habitat.get_config("configs/tasks/expl.yaml")
    with habitat.Env(config=config) as env:
        import pdb; pdb.set_trace()
        observations = env.reset()
        count_steps = 0
        while not env.episode_over:
            observations = env.step(env.action_space.sample())
            count_steps += 1
        print("Episode finished after {} steps.".format(count_steps))

if __name__ == "__main__":
    expl()

