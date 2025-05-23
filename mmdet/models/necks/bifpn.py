import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

from mmdet.registry import MODELS
from mmcv.cnn import ConvModule

@MODELS.register_module()
class BIFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(BIFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = act_cfg
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.stack = stack

        if end_level == -1:
            self.backbone_end_level = self.num_ins#4
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level#-1
        self.add_extra_convs = add_extra_convs#F
        self.extra_convs_on_inputs = extra_convs_on_inputs#T

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):#只是調整維度
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for ii in range(stack):
            self.stack_bifpn_convs.append(BiFPNModule(channels=out_channels,
                                                      levels=self.backbone_end_level-self.start_level,
                                                      conv_cfg=conv_cfg,
                                                      norm_cfg=norm_cfg,
                                                      activation=self.activation))
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
     for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)  # Use PyTorch's xavier uniform initialization
            if m.bias is not None:
                init.constant_(m.bias, 0)  # Initialize bias to zero if exists

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # part 1: build top-down and down-top path with stack
        used_backbone_levels = len(laterals)
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,  # out channels
                 levels,  # 4
                 init=0.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 eps=1e-4):
        super(BiFPNModule, self).__init__()
        self.activation = activation
        self.eps = eps
        self.levels = levels
        self.bifpn_convs = nn.ModuleList()

        # Weighted parameters
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))  # (2,4)
        self.relu1 = nn.ReLU(inplace=False)
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))  # (3,2)
        self.relu2 = nn.ReLU(inplace=False)

        # Create convs for top-down and down-top paths
        for _ in range(2):  # 2 passes
            for _ in range(levels - 1):
                fpn_conv = nn.Sequential(
                    ConvModule(
                        channels,
                        channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=self.activation,
                        inplace=False
                    )
                )
                self.bifpn_convs.append(fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == self.levels

        levels = self.levels
        # Detach to avoid in-place modification errors in autograd
        w1 = self.relu1(self.w1.detach().clone())
        w1 = w1 / (torch.sum(w1, dim=0, keepdim=True) + self.eps)

        w2 = self.relu2(self.w2.detach().clone())
        w2 = w2 / (torch.sum(w2, dim=0, keepdim=True) + self.eps)


        idx_bifpn = 0
        # Clone inputs to avoid modifying original tensors inplace
        inputs_clone = [x.clone() for x in inputs]

        # Start top-down path, initialize with cloned inputs
        top_down_feats = inputs_clone.copy()

        for i in range(levels - 1, 0, -1):
            td = (w1[0, i - 1] * top_down_feats[i - 1] +
                  w1[1, i - 1] * F.interpolate(top_down_feats[i], scale_factor=2, mode='nearest'))
            td = td / (w1[0, i - 1] + w1[1, i - 1] + self.eps)
            top_down_feats[i - 1] = self.bifpn_convs[idx_bifpn](td)
            idx_bifpn += 1

        # Start down-top path, copy top-down features
        down_top_feats = top_down_feats.copy()

        for i in range(levels - 2):
            dt = (w2[0, i] * down_top_feats[i + 1] +
                  w2[1, i] * F.max_pool2d(down_top_feats[i], kernel_size=2) +
                  w2[2, i] * inputs_clone[i + 1])
            dt = dt / (w2[0, i] + w2[1, i] + w2[2, i] + self.eps)
            down_top_feats[i + 1] = self.bifpn_convs[idx_bifpn](dt)
            idx_bifpn += 1

        # Last level update
        last = (w1[0, levels - 1] * down_top_feats[levels - 1] +
                w1[1, levels - 1] * F.max_pool2d(down_top_feats[levels - 2], kernel_size=2))
        last = last / (w1[0, levels - 1] + w1[1, levels - 1] + self.eps)
        down_top_feats[levels - 1] = self.bifpn_convs[idx_bifpn](last)

        return down_top_feats