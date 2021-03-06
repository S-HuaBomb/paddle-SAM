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

from argparse import Namespace
import os
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import paddle
from paddle.io import DataLoader

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im, log_image
from options.test_options import TestOptions
from models.psp import pSp


def run():
	test_opts = TestOptions().parse()

	out_path_results = os.path.join(test_opts.exp_dir, 'inference_side_by_side')
	os.makedirs(out_path_results, exist_ok=True)

	# update test options with options used during training
	ckpt = paddle.load(test_opts.checkpoint_path)
	opts = ckpt['opts']
	del ckpt
	opts.update(vars(test_opts))
	print(opts)
	opts = Namespace(**opts)
	net = pSp(opts)
	
	net.eval()

	age_transformers = [AgeTransformer(target_age=age) for age in opts.target_age.split(',')]

	print(f'Loading dataset for {opts.dataset_type}')
	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	dataset = InferenceDataset(root=opts.data_path,
							   transform=transforms_dict['transform_inference'],
							   opts=opts,
							   return_path=True)
	dataloader = DataLoader(dataset,
							batch_size=opts.test_batch_size,
							shuffle=False,
							num_workers=int(opts.test_workers),
							drop_last=False)

	if opts.n_images is None:
		opts.n_images = len(dataset)

	global_time = []
	global_i = 0
	for input_batch, image_paths in tqdm(dataloader):
		if global_i >= opts.n_images:
			break
		batch_results = {}
		for idx, age_transformer in enumerate(age_transformers):
			with paddle.no_grad():
				input_age_batch = [age_transformer(img) for img in input_batch]
				input_age_batch = paddle.stack(input_age_batch)
				input_cuda = paddle.to_tensor(input_age_batch, paddle.float32)
				tic = time.time()
				result_batch = run_on_batch(input_cuda, net, opts)
				toc = time.time()
				global_time.append(toc - tic)

				resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
				for i in range(len(input_batch)):
					result = tensor2im(result_batch[i])
					im_path = image_paths[i]
					input_im = log_image(input_batch[i], opts)
					if im_path not in batch_results.keys():
						batch_results[im_path] = np.array(input_im.resize(resize_amount))
					batch_results[im_path] = np.concatenate([batch_results[im_path],
															 np.array(result.resize(resize_amount))],
															axis=1)

		for im_path, res in batch_results.items():
			image_name = os.path.basename(im_path)
			im_save_path = os.path.join(out_path_results, image_name)
			Image.fromarray(np.array(res)).save(im_save_path)
			global_i += 1


def run_on_batch(inputs, net, opts):
	result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
	return result_batch


if __name__ == '__main__':
	run()
