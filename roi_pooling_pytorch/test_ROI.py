import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from roi_pooling.functions.roi_pooling import roi_pooling_2d
from roi_pooling.functions.roi_pooling import roi_pooling_2d_pytorch
import pdb 

batch_size = 3
n_channels = 4
input_size = (12, 8)
output_size = (5, 7)
spatial_scale = 0.6
x_np = np.arange(batch_size * n_channels * input_size[0] * input_size[1], dtype=np.float32)
x_np = x_np.reshape((batch_size, n_channels, *input_size))
np.random.shuffle(x_np)
x = torch.from_numpy(2 * x_np / x_np.size - 1)
rois = torch.FloatTensor([
    [0, 1, 1, 6, 6],
    [2, 6, 2, 7, 11],
    [1, 3, 1, 5, 10],
    [0, 3, 3, 3, 3]
])

n_rois = rois.shape[0]
x = x.cuda()  ## torch.Size([3, 4, 12, 8])
rois = rois.cuda()   ## torch.Size([4, 5]) 

# pdb.set_trace() 

y_var = roi_pooling_2d(x, rois, output_size, spatial_scale=spatial_scale)
## torch.Size([4, 4, 5, 7])
