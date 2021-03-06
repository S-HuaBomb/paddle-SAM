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

import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import paddle
from PIL import Image
from paddle.io import DataLoader
import sys

# import paddle.distributed as dist

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im, log_image
from options.test_options import TestOptions
from models.psp import pSp

# dist.init_parallel_env()


def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = paddle.load(test_opts.checkpoint_path)
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    print(opts)
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()

    age_transformers = [AgeTransformer(target_age=age) for age in opts.target_age.split(',')]

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_time = []
    for age_transformer in age_transformers:
        print(f"Running on target age: {age_transformer.target_age}")
        global_i = 0
        for input_batch in tqdm(dataloader):
            if global_i >= opts.n_images:
                break
            with paddle.no_grad():
                input_age_batch = [age_transformer(img) for img in input_batch]
                input_age_batch = paddle.stack(input_age_batch)
                input_age_batch = paddle.to_tensor(input_age_batch, paddle.float32)
                tic = time.time()
                result_batch = run_on_batch(input_age_batch, net, opts)
                toc = time.time()
                global_time.append(toc - tic)

                for i in range(len(input_batch)):
                    result = tensor2im(result_batch[i])
                    im_path = dataset.paths[global_i]

                    if opts.couple_outputs or global_i % 100 == 0:
                        input_im = log_image(input_batch[i], opts)
                        resize_amount = (256, 256) if opts.resize_outputs else (1024, 1024)
                        res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                              np.array(result.resize(resize_amount))], axis=1)
                        Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

                    age_out_path_results = os.path.join(out_path_results, age_transformer.target_age)
                    os.makedirs(age_out_path_results, exist_ok=True)
                    im_save_path = os.path.join(age_out_path_results, os.path.basename(im_path))
                    Image.fromarray(np.array(result.resize(resize_amount))).save(im_save_path)
                    global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def run_on_batch(inputs, net, opts):
    result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    return result_batch


if __name__ == '__main__':
    run()
