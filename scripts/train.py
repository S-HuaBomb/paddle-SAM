#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint


sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach_aging import Coach
# import paddle.distributed as dist


def main():
	# dist.init_parallel_env()
	opts = TrainOptions().parse()
	if os.path.exists(opts.exp_dir):
		raise Exception('Options exp_dir ... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	main()
	# dist.spawn(main, args=(True,))
	# dist.spawn(main)