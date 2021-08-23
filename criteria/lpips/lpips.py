import paddle
import paddle.nn as nn
from configs.paths_config import model_paths

from criteria.lpips.networks import get_network, LinLayers


class LPIPS(nn.Layer):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.set_state_dict(paddle.load(model_paths['lin_alex0.1']))


    def forward(self, x: paddle.Tensor, y: paddle.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin.model.sublayers())]

        return paddle.sum(paddle.concat(res, 0)) / x.shape[0]
