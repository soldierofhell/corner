import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import train_cfg
from ..box_utils import match, log_sum_exp

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, neg_pos, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = train_cfg['variance']

    def forward(self, predictions, targets, segs):
        loc_data, conf_data, priors, seg_data = predictions
        # loc_data: offset branch, [batch_size, \sum_k w_kxh_kxk=120272, q=4, offsets=4]
        # conf_data: score branch, [batch_size, \sum_k w_kxh_kxk=120272, q=4, scores=2]) 
        # priors: default boxes [\sum_k w_kxh_kxk=120272, x_1,x_2,s_x1,s_x2=4]
        # seg_data: position sensitive segmentation [batch_size, w*h*g*g, 1]
        
        # targets: corners gt, list([batch_size][q=4][n, [x1,x2,y1,y2,label]])
        # segs: segmentation gt, [batch_size, w, h, g*g]
        
        num = loc_data.size(0) # batch_size
        priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.zeros(num, num_priors, 4, 4).float()
        conf_t = torch.zeros(num, num_priors, 4).long()
        for idx in range(num):
            ## match top_left
            for idx_idx in range(4):
                truths = targets[idx][idx_idx][:, :4].data
                labels = targets[idx][idx_idx][:, -1].data
                defaults = priors.data
                
                match(self.threshold, truths, defaults, self.variance, labels,
                      loc_t, conf_t, idx, idx_idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        # conf_t: [batch_size, \sum_k w_kxh_kxk=120272, q=4]

        pos = conf_t > 0
        # pos: [batch_size, \sum_k w_kxh_kxk=120272, q=4]
        
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # pos_idx, [8, 120272, 4, 4]
        
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # batch_conf: [batch_size * (\sum_k w_k * h_k * k) * q, 2]

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))        
        # loss_c: [batch_size * (\sum_k w_k * h_k * k) * q, 1]
        
        # Hard Negative Mining
        loss_c[pos.view(-1,1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1) # [batch_size, (\sum_k w_k * h_k * k) * q]
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        pos = pos.view(num, -1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        conf_data_v = conf_data.view(num, -1, self.num_classes)
        conf_t_v = conf_t.view(num, -1)

        pos_idx = pos.unsqueeze(2).expand_as(conf_data_v)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data_v)
        conf_p = conf_data_v[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t_v[(pos+neg).gt(0)]
        
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
            
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N

        ## L_seg - Dice loss
        eps = 1e-5
        seg_gt = segs.view(-1, 1)
        intersection = torch.sum(seg_data*seg_gt)
        union = torch.sum(seg_data + seg_gt) + eps
        loss_s = 1 - 2.0*intersection/union

        return loss_l, loss_c, loss_s
