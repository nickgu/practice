#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torch

if __name__=='__main__':
    cuda = torch.device('cuda')     # Default CUDA device

    x = torch.tensor([1., 2.], device=cuda)
    # x.device is device(type='cuda', index=0)
    y = torch.tensor([1., 2.], device=cuda)
    # y.device is device(type='cuda', index=0)

    # allocates a tensor on GPU 1
    a = torch.tensor([1., 2.], device=cuda)

    # transfers a tensor from CPU to GPU 1
    b = torch.tensor([1., 2.]).cuda()
    # a.device and b.device are device(type='cuda', index=1)

    # You can also use ``Tensor.to`` to transfer a tensor:
    b2 = torch.tensor([1., 2.]).to(device=cuda)
    # b.device and b2.device are device(type='cuda', index=1)

    c = a + b
    # c.device is device(type='cuda', index=1)

    z = x + y
    # z.device is device(type='cuda', index=0)

    # even within a context, you can specify the device
    # (or give a GPU index to the .cuda call)
    d = torch.randn(2, device=cuda)
    e = torch.randn(2).to(cuda)
    f = torch.randn(2).cuda(cuda)
    # d.device, e.device, and f.device are all device(type='cuda', index=2)

