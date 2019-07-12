import torch
from torch.utils.cpp_extension import load

rpsroi_pooling = load(name="rpsroi_pooling", sources=["../cpu/rpsroi_pooling.cpp"])


class RPSRoIPoolingFunction():
    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)
        assert(self.pooled_width == self.pooled_height)
        assert(self.pooled_height == self.group_size)

    def __call__(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        output = rpsroi_pooling.RPSROIPool_forward_cpu(
            self.pooled_height,
            self.pooled_width,
            self.spatial_scale,
            self.group_size,
            self.output_dim,
            features,
            rois);

        return output
