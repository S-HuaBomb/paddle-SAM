import paddle


def normalize_activation(x, eps=1e-10):
    norm_factor = paddle.sqrt(paddle.sum(x ** 2, 1, keepdim=True))
    return x / (norm_factor + eps)

