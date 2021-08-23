"""
# sam的模型转换
def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

from models.psp import pSp
import torch
import paddle
from argparse import Namespace

input_fp = "pretrained_models/sam_ffhq_aging.pt"
output_fp = "pretrained_models/sam_ffhq_aging.pdparams"

torch_dict = torch.load(input_fp, map_location=torch.device('cpu'))

print(torch_dict.keys())
encoder_torch = get_keys(torch_dict,'encoder')
# decoder_torch = get_keys(torch_dict,'decoder')
pre_encoder_torch = get_keys(torch_dict,'pretrained_encoder')
opts = torch_dict['opts']
latent_avg = torch_dict['latent_avg']
del torch_dict

# update the training options
opts['checkpoint_path'] = None
opts['pretrained_psp_path'] = "pretrained_models/psp_ffhq_encode.pdparams"
opts['stylegan_weights'] = "pretrained_models/stylegan2-ffhq-config-f.pdparams"
opts['ir_se50'] = 'pretrained_models/model_ir_se50.pdparams'
opts = Namespace(**opts)
print(opts)


net = pSp(opts)
encoder_paddle = net.encoder.state_dict()
encoder_list_paddle = list(encoder_paddle.keys())
encoder_list_torch = list(encoder_torch.keys())

pre_encoder_paddle = net.pretrained_encoder.state_dict()
pre_encoder_list_paddle = list(pre_encoder_paddle.keys())
pre_encoder_list_torch = list(pre_encoder_torch.keys())

i = 0
j = 0
while i < len(encoder_list_torch):
    key = encoder_list_torch[i]
    if 'num' in key:
        i = i + 1
        continue
    weight = encoder_torch[key].detach().cpu().numpy()
    if 'linear.weight' in key:
        print("weight {} need to be trans".format(key))
        weight = weight.transpose()
    encoder_paddle[encoder_list_paddle[j]] = weight
    i = i + 1
    j = j + 1

# test loading encoder weight
net.encoder.set_state_dict(encoder_paddle)
del encoder_torch

i = 0
j = 0
while i < len(pre_encoder_list_torch):
    key = pre_encoder_list_torch[i]
    if 'num' in key:
        i = i + 1
        continue
    weight = pre_encoder_torch[key].detach().cpu().numpy()
    if 'linear.weight' in key:
        print("weight {} need to be trans".format(key))
        weight = weight.transpose()
    pre_encoder_paddle[pre_encoder_list_paddle[j]] = weight
    i = i + 1
    j = j + 1

# test loading pretrained_encoder weight
net.pretrained_encoder.set_state_dict(pre_encoder_paddle)
del pre_encoder_torch

decoder_paddle = net.decoder.state_dict()

del net

state_dict_paddle = {
    'encoder': encoder_paddle,
    'decoder': decoder_paddle,
    'pretrained_encoder': pre_encoder_paddle
}

del encoder_paddle
del decoder_paddle
del pre_encoder_paddle

save_dict = {
	'state_dict': state_dict_paddle,
	'opts': vars(opts),
    "latent_avg": latent_avg.cpu().numpy()
}
paddle.save(save_dict, output_fp)"""


# age classifier的模型转换
from models.dex_vgg import VGG
import torch
import paddle
from argparse import Namespace

input_fp = "pretrained_models/dex_age_classifier.pth"
output_fp = "pretrained_models/dex_age_classifier.pdparams"

torch_dict = torch.load(input_fp, map_location=torch.device('cpu'))

print(torch_dict.keys())
state_dict_torch = torch_dict['state_dict']

net = VGG()
state_dict_paddle = net.state_dict()
print(state_dict_paddle.keys())
state_dict_keys = state_dict_torch.keys()
for key in state_dict_paddle.keys():
    k = key

    if 'fc' in key and 'weight' in key: # 全转化为list，不然一个list，一个tuple，就不相等了。
        state_dict_paddle[key]=state_dict_torch[k].T.cpu().numpy()
    else:
        state_dict_paddle[key]=state_dict_torch[k].cpu().numpy()

#net.set_state_dict(state_dict_paddle)

#print(net.state_dict().keys())
#paddle.save(net.state_dict(), output_fp)
paddle.save(state_dict_paddle, output_fp)
