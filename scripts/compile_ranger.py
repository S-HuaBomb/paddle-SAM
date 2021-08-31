import paddle
from paddle.utils.cpp_extension import load

custom_ops = load(
        name='ranger_op',
        sources=["training/ranger_cuda.cc", "training/ranger_cuda.cu"],
        verbose=True)
