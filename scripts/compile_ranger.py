import paddle
from paddle.utils.cpp_extension import load

custom_ops = load(
        name='ranger_op',
        sources=["work/training/ranger_cuda.cc", "work/training/ranger_cuda.cu"],
        verbose=True)
