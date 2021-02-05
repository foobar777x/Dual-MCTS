"""
Created on Tue 06/01/2020 12:07:20 2020
This piece of software is bound by The Apache License
Copyright (c) 2019 Prashank Kadam
Code written by : Prashank Kadam
User name - prashank
Email ID : kadam.pr@husky.neu.edu
version : 1.0
"""
# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Simple Multiple Policy-Value MCTS nim example.

Take a look at the log-learner.txt in the output directory.

If you want more control, check out `mpv.py`.
"""

from absl import app
from absl import flags

from open_spiel.python.algorithms.mpv_mcts import mpv
from open_spiel.python.utils import spawn

import pudb

flags.DEFINE_string("path", None, "Where to save checkpoints.")
FLAGS = flags.FLAGS


def main(unused_argv):
  config = mpv.Config(
      game="nim",
      path=FLAGS.path,

      learning_rate_1=0.01,
      weight_decay_1=1e-4,
      train_batch_size_1=128,
      replay_buffer_size_1=2**14,
      replay_buffer_reuse_1=4,
      max_steps_1=20,
      checkpoint_freq_1=5,

      actors_1=4,
      evaluators_1=4,
      uct_c_1=1,                  # Exploration term
      max_simulations_1=20,
      policy_alpha_1=0.25,
      policy_epsilon_1=1,
      temperature_1=1,
      temperature_drop_1=4,
      evaluation_window_1=50,
      eval_levels_1=7,

      nn_model_1="resnet",
      nn_width_1=8,
      nn_depth_1=4,
      observation_shape_1=None,
      output_size_1=None,

      learning_rate_2=0.01,
      weight_decay_2=1e-4,
      train_batch_size_2=128,
      replay_buffer_size_2=2 ** 14,
      replay_buffer_reuse_2=4,
      max_steps_2=20,
      checkpoint_freq_2=5,

      actors_2=4,
      evaluators_2=4,
      uct_c_2=0.10,                   # Exploitation term
      max_simulations_2=5,
      policy_alpha_2=0.25,
      policy_epsilon_2=10,
      temperature_2=0.2,
      temperature_drop_2=4,
      evaluation_window_2=50,
      eval_levels_2=7,

      nn_model_2="resnet",
      nn_width_2=8,
      nn_depth_2=4,
      observation_shape_2=None,
      output_size_2=None,

      quiet=True,
  )

  # pudb.set_trace()

  mpv.mpv(config)


if __name__ == "__main__":
  with spawn.main_handler():
    app.run(main)
