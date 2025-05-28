# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import copy
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from torchvision.ops import nms

from mmdet.structures.bbox import HorizontalBoxes, BaseBoxes 
from mmdet.structures import DetDataSample

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList
from ..task_modules.prior_generators import MlvlPointGenerator
from ..task_modules.assigners import PointAssigner, MaxIoUAssigner
from ..task_modules.samplers import PseudoSampler
from ..utils import (filter_scores_and_topk, images_to_levels, multi_apply)
from .anchor_free_head import AnchorFreeHead

def get_pad_shape(img_meta):
        if hasattr(img_meta, 'get'):
            pad_shape = img_meta.get('pad_shape', None)
        elif hasattr(img_meta, '__getitem__'):
            pad_shape = img_meta['pad_shape'] if 'pad_shape' in img_meta else None
        elif hasattr(img_meta, 'metainfo') and isinstance(img_meta.metainfo, dict):
            pad_shape = img_meta.metainfo.get('pad_shape', None)
        else:
            pad_shape = None

        if pad_shape is None:
            # Optionally fallback to img_shape if pad_shape missing
            if hasattr(img_meta, 'get'):
                pad_shape = img_meta.get('img_shape', None)
            elif hasattr(img_meta, '__getitem__'):
                pad_shape = img_meta['img_shape'] if 'img_shape' in img_meta else None
            elif hasattr(img_meta, 'metainfo') and isinstance(img_meta.metainfo, dict):
                pad_shape = img_meta.metainfo.get('img_shape', None)

        if pad_shape is None:
            raise ValueError(f"pad_shape and img_shape are both missing in img_meta: {img_meta}")

        return pad_shape

def multi_apply(func, *args, **kwargs):
        pfunc = lambda arg: func(*arg, **kwargs) if isinstance(arg, (tuple, list)) else func(arg, **kwargs)
        map_results = map(pfunc, zip(*args))
        results = tuple(map(list, zip(*map_results)))
    
        return results


@MODELS.register_module()
class RepPointsHead(AnchorFreeHead):
    """RepPoint head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        point_feat_channels (int): Number of channels of points features.
        num_points (int): Number of points.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Sequence[int]): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox_init (:obj:`ConfigDict` or dict): Config of initial points
            loss.
        loss_bbox_refine (:obj:`ConfigDict` or dict): Config of points loss in
            refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 train_cfg=None,
                 point_feat_channels: int = 256,
                 num_points: int = 9,
                 gradient_mul: float = 0.1,
                 point_strides: Sequence[int] = [8, 16, 32, 64],
                 point_base_scale: int = 4,
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init: ConfigType = dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine: ConfigType = dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 use_grid_points: bool = False,
                 center_init: bool = True,
                 transform_method: str = 'moment',
                 moment_mul: float = 0.01,
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='reppoints_cls_out',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.use_grid_points = use_grid_points
        self.center_init = center_init
        self.num_classes = num_classes
        #print(f"[DEBUG] RepPointsRPNHead initialized with num_classes={self.num_classes}")
        
        # we use deform conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            train_cfg=train_cfg,  # add this explicitly!
            loss_cls=loss_cls,
            init_cfg=init_cfg,
            **kwargs)
        
        self._init_layers()

        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.prior_generator = MlvlPointGenerator(self.point_strides, offset=0.)

        # print(f"num_points: {self.num_points}")
        # print(f"strides: {self.point_strides}")
        # print(f"point_base_scale: {self.point_base_scale}")


        if self.train_cfg:
            self.init_assigner = TASK_UTILS.build(
                self.train_cfg['init']['assigner'])
            self.refine_assigner = TASK_UTILS.build(
                self.train_cfg['refine']['assigner'])

            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', True) # False

        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        self.loss_bbox_init = MODELS.build(loss_bbox_init)
        self.loss_bbox_refine = MODELS.build(loss_bbox_refine)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            
        #print(f"[_init_layers] num_points: {self.num_points}")
        pts_out_dim = 2 * self.num_points

        self.reppoints_cls_conv = DeformConv2d(self.feat_channels,
                                               self.point_feat_channels,
                                               self.dcn_kernel, 1,
                                               self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv2d(self.feat_channels,
                                                      self.point_feat_channels,
                                                      self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)
    
    def predict_raw(self, x):
        out = multi_apply(self.forward_single, x, self.point_strides)
        return out

    def points2bbox(self, pts: Tensor, y_first: bool = True) -> Tensor:
        """Convert points to bbox with proper reshaping."""
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)
        
        pts = pts.view(-1, self.num_points, 2)
        pts_y = pts[:, :, 0] if y_first else pts[:, :, 1]
        pts_x = pts[:, :, 1] if y_first else pts[:, :, 0]
        
        if self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            half_width = pts_x_std * torch.exp(moment_transfer[0])
            half_height = pts_y_std * torch.exp(moment_transfer[1])
            bbox = torch.cat([
                pts_x_mean - half_width, 
                pts_y_mean - half_height,
                pts_x_mean + half_width, 
                pts_y_mean + half_height
            ], dim=1)
        else:
            raise NotImplementedError
            
        return bbox.squeeze()

    def gen_grid_from_reg(self, reg: Tensor, previous_boxes: Tensor) -> Tuple[Tensor]:
        B, _, H, W = reg.shape
        # Reshape to (B, num_points, 4, H, W)

        reg = reg.view(B, self.num_points, 4, H, W)
        reg_mean = reg[:, :, :2, :, :]  # (B, num_points, 2, H, W)
        reg_std = reg[:, :, 2:, :, :]   # (B, num_points, 2, H, W)

        previous_boxes = previous_boxes.view(B, self.num_points, 4, H, W) # if this is working, uncomment this
        # previous_boxes = previous_boxes.unsqueeze(1).expand(-1, self.num_points, -1, -1, -1)

        # Extract x1, y1, x2, y2
        x1 = previous_boxes[:, :, 0, :, :]
        y1 = previous_boxes[:, :, 1, :, :]
        x2 = previous_boxes[:, :, 2, :, :]
        y2 = previous_boxes[:, :, 3, :, :]

        # Compute box center and size for each point
        bxy = torch.stack([(x1 + x2) / 2., (y1 + y2) / 2.], dim=2)  # (B, num_points, 2, H, W)
        bwh = torch.stack([(x2 - x1).clamp(min=1e-6), (y2 - y1).clamp(min=1e-6)], dim=2)  # (B, num_points, 2, H, W)

        grid_topleft = bxy + bwh * reg_mean - 0.5 * bwh * torch.exp(reg_std)  # (B, num_points, 2, H, W)
        grid_wh = bwh * torch.exp(reg_std)  # (B, num_points, 2, H, W)

        grid_left = grid_topleft[:, :, 0, :, :]  # (B, num_points, H, W)
        grid_top = grid_topleft[:, :, 1, :, :]   # (B, num_points, H, W)
        grid_width = grid_wh[:, :, 0, :, :]      # (B, num_points, H, W)
        grid_height = grid_wh[:, :, 1, :, :]     # (B, num_points, H, W)

        intervel = torch.linspace(0., 1., self.dcn_kernel, device=reg.device).view(
            1, self.dcn_kernel, 1, 1)

        # Grid x
        grid_x = grid_left.unsqueeze(2) + grid_width.unsqueeze(2) * intervel.unsqueeze(1)  # (B, num_points, dcn_kernel, H, W)
        grid_x = grid_x.view(B, -1, H, W)  # flatten point * kernel

        # Grid y
        grid_y = grid_top.unsqueeze(2) + grid_height.unsqueeze(2) * intervel.unsqueeze(1)  # (B, num_points, dcn_kernel, H, W)
        grid_y = grid_y.view(B, -1, H, W)

        # Stack to get (B, 2*P*K, H, W)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)  # (B, num_points, 2, dcn_kernel, H, W)
        grid_yx = grid_yx.view(B, -1, H, W)

        # Regressed box = [x1, y1, x2, y2] per point
        regressed_bbox = torch.cat([
            grid_left.view(B, -1, H, W),
            grid_top.view(B, -1, H, W),
            (grid_left + grid_width).view(B, -1, H, W),
            (grid_top + grid_height).view(B, -1, H, W)
        ], dim=1)

        return grid_yx, regressed_bbox

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor]:
        return multi_apply(self.forward_single, feats, self.point_strides) # pass strides
    
    def bbox_to_points(self, bbox: Tensor) -> Tensor:
        x1 = bbox[:, 0:1]  # (B, 1, H, W)
        y1 = bbox[:, 1:2]
        x2 = bbox[:, 2:3]
        y2 = bbox[:, 3:4]

        w = x2 - x1
        h = y2 - y1
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        xs = torch.cat([x1, x_center, x2], dim=1)  # (B, 3, H, W)
        ys = torch.cat([y1, y_center, y2], dim=1)  # (B, 3, H, W)

        # Generate grid points: 9 points (3x3)
        grid_points = []
        for yi in range(3):
            for xi in range(3):
                grid_points.append(xs[:, xi:xi+1])  # x
                grid_points.append(ys[:, yi:yi+1])  # y
                grid_points.append(w)               # width (broadcasted)
                grid_points.append(h)               # height (broadcasted)

        points = torch.cat(grid_points, dim=1)  # (B, 36, H, W)
        return points
    
    def forward_single(self, x: Tensor, stride: int) -> Tuple[Tensor]:
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)

        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 4 # Reduced from /2 to /4 for smaller init boxes /8 working
            points_init = dcn_base_offset / dcn_base_offset.max() * scale * stride # remove if not working
            bbox_init = x.new_tensor([-scale, -scale, scale, scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
            bbox_init = None  # Will be assigned dummy tensor later

        cls_feat = x
        pts_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)

        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))

        if self.use_grid_points:
            B, _, H, W = pts_out_init.shape  # B, 18, H, W
            bbox_init = bbox_init.expand(B, 4, H, W)
            bbox_init = self.bbox_to_points(bbox_init)
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
            bbox_out_init = pts_out_init
        
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul[:, :1, :, :] - dcn_base_offset

        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))

        #print(f"[DEBUG] cls_out.shape = {cls_out.shape}")

        if self.use_grid_points:
            bbox_out_init = bbox_out_init.expand(B, 36, H, W)
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()
            bbox_out_refine = pts_out_refine
        
        if self.training:
            return cls_out, pts_out_init, pts_out_refine, bbox_out_init, bbox_out_refine
        else:
            return torch.sigmoid(cls_out), pts_out_init, pts_out_refine, bbox_out_init, bbox_out_refine

        # THIS is WORKING
        # if self.training:
        #     return cls_out, pts_out_init, pts_out_refine, bbox_out_init, bbox_out_refine
        # else:
        #     return cls_out, pts_out_refine, bbox_out_init, bbox_out_refine

    def centers_to_bboxes_single(self, centers: Tensor) -> Tensor:
        """
        Convert center points [N, 2] into bounding boxes [N, 4] for a single image.
        """
        # centers: [N, 2] assumed to be (x_center, y_center)
        # You need to create a bbox around the center with some predefined width/height or offsets

        wh = torch.full_like(centers, fill_value=1.0)  # dummy width/height, modify if needed
        xy1 = centers - wh / 2  # top-left
        xy2 = centers + wh / 2  # bottom-right
        bboxes = torch.cat([xy1, xy2], dim=-1)  # [N, 4]

        return bboxes

    def centers_to_bboxes(self, point_list: List[Tensor]) -> List[List[Tensor]]:
        bbox_list = []
        for point in point_list:
            bbox = []
            # 🔥 Fix: squeeze batch dimension if it exists
            if point.ndim == 3 and point.shape[0] == 1:
                point = point.squeeze(0)  # from [1, N, 2] → [N, 2]

            if point.ndim != 2 or point.shape[1] < 2:
                raise ValueError(f"Expected shape [N, 2+] for point, got {point.shape}")

            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.tensor([-scale, -scale, scale, scale], device=point.device).view(1, 4)

                coords = point[:, :2]
                bbox_center = coords.repeat(1, 2)  # [N, 4]
                bbox.append(bbox_center + bbox_shift)

            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list: List[Tensor],
                      pred_list: List[Tensor]) -> List[Tensor]:
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).reshape(-1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list
    
    def _get_targets_single(self,
                        flat_proposals: Tensor,
                        valid_flags: Tensor,
                        gt_instances: InstanceData,
                        gt_instances_ignore: Optional[InstanceData] = None,
                        stage: str = 'init',
                        unmap_outputs: bool = True) -> tuple:

        # Ensure valid_flags is a 1D boolean tensor
        inside_flags = valid_flags if isinstance(valid_flags, Tensor) else torch.cat(valid_flags, dim=0)
        inside_flags = inside_flags.to(torch.bool).reshape(-1)

        # Ensure flat_proposals is [num_proposals, 4]
        if flat_proposals.dim() == 3:
            flat_proposals = flat_proposals.squeeze(0)
        elif flat_proposals.dim() == 2 and flat_proposals.size(0) == 1:
            flat_proposals = flat_proposals.view(-1, 4)

        if flat_proposals.size(-1) != 4:
            raise ValueError(f"Expected proposals of shape [N, 4], got {flat_proposals.shape}")

        if not inside_flags.any():
            raise ValueError('No valid proposals inside image boundary')

        # Get valid proposals
        proposals = flat_proposals[inside_flags, :]

        # Select assigner and pos_weight
        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg['init']['pos_weight']
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg['refine']['pos_weight']

        pred_instances = InstanceData()

        if isinstance(assigner, PointAssigner):
            # Convert boxes to center points
            points = torch.stack([
                (proposals[:, 0] + proposals[:, 2]) / 2,
                (proposals[:, 1] + proposals[:, 3]) / 2
            ], dim=-1)

            #print("Sample proposals (first 5):", proposals[:5])

            total_points = points.size(0)
            num_levels = len(self.point_strides)

            assert total_points % num_levels == 0, (
                f"Total points ({total_points}) must be divisible by number of levels ({num_levels})")
            
            pts_per_level = total_points // num_levels
            num_points_per_level = [pts_per_level] * num_levels

            stride_list = [torch.full((n,), s, device=points.device)
                        for s, n in zip(self.point_strides, num_points_per_level)]
            stride_tensor = torch.cat(stride_list, dim=0)

            assert stride_tensor.size(0) == points.size(0), "Stride tensor size mismatch"

            proposals_with_stride = torch.cat([points, stride_tensor[:, None]], dim=1)
            pred_instances.priors = proposals_with_stride

        elif isinstance(assigner, MaxIoUAssigner):
            if proposals.size(-1) == 4:
                pred_instances.bboxes = HorizontalBoxes(proposals)
            else:
                raise ValueError("MaxIoUAssigner expects bbox format proposals with 4 coordinates")

        else:
            raise TypeError(f"Unsupported assigner type: {type(assigner)}")
        
        # print("pred_instances keys:", pred_instances.__dict__.keys())
        # print("proposals_with_stride (sample):", pred_instances.priors[:5])
        # print("GT boxes (sample):", gt_instances.bboxes.tensor[:5])

        # Assign and sample
        assign_result = assigner.assign(pred_instances, gt_instances, gt_instances_ignore)

        # Ensure labels exist in assign_result
        if assign_result.labels is None:
            assign_result.labels = torch.full_like(assign_result.gt_inds, 0)
        
        sampler = PseudoSampler(context=self) if stage == 'init' else self.sampler
        sampling_result = sampler.sample(assign_result, pred_instances, gt_instances)

        # Prepare target tensors
        num_valid = proposals.size(0)
        bbox_gt = proposals.new_zeros((num_valid, 4))
        pos_proposals = proposals.new_zeros((num_valid, 4))
        proposals_weights = proposals.new_zeros((num_valid, 4))
        labels = proposals.new_full((num_valid,), self.num_classes, dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # print("sampling_result.pos_inds:", sampling_result.pos_inds)
        # print("sampling_result.neg_inds:", sampling_result.neg_inds)

        if len(pos_inds) > 0:
            # print("if pos_inds > 0: labels at pos_inds:", labels[pos_inds])
            # print(f"pos_inds: {pos_inds}")
            # print(f"sampling_result.pos_gt_bboxes.tensor device: {sampling_result.pos_gt_bboxes.tensor.device}")
            # print(f"bbox_gt device: {bbox_gt.device}")
            # print(f"sampling_result.pos_gt_bboxes.tensor shape: {sampling_result.pos_gt_bboxes.tensor.shape}")
            # print(f"sampling_result.pos_gt_bboxes.tensor sample: {sampling_result.pos_gt_bboxes.tensor[:5]}")
            bbox_gt[pos_inds] = sampling_result.pos_gt_bboxes.tensor
            pos_proposals[pos_inds] = proposals[pos_inds]
            proposals_weights[pos_inds] = 1.0
            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.num_classes == 1:
                labels[labels == 1] = 0
            label_weights[pos_inds] = pos_weight if pos_weight > 0 else 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        
        def unmap(data: Tensor, count: int, inds: Tensor, fill=0):
            if inds.dtype == torch.bool:
                inds = inds.nonzero(as_tuple=False).squeeze(1)
            if data.dim() == 1:
                ret = data.new_full((count,), fill)
                ret[inds] = data
            else:
                new_shape = (count,) + data.shape[1:]
                ret = data.new_full(new_shape, fill)
                ret[inds, :] = data
            return ret
        
        # print("labels before unmap:", labels.unique())
        # print("pos_inds:", pos_inds)
        # print("labels at pos_inds:", labels[pos_inds])
        # print("inside_flags sum:", inside_flags.sum())

        if unmap_outputs:
            num_total = flat_proposals.size(0)
            inside_inds = inside_flags.nonzero(as_tuple=False).squeeze(1)
            # Ensure we preserve the positive labels

            # print("inside_inds shape:", inside_inds.shape)
            # print("pos_inds_unmapped:", inside_inds[pos_inds])

            # Unmap data first
            labels = unmap(labels, num_total, inside_flags, fill=-1)
            label_weights = unmap(label_weights, num_total, inside_flags, fill=0)
            bbox_gt = unmap(bbox_gt, num_total, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total, inside_flags)
            proposals_weights = unmap(proposals_weights, num_total, inside_flags)

            # Then get unmapped pos/neg indices
            pos_inds_unmapped = inside_inds[pos_inds]
            neg_inds_unmapped = inside_inds[neg_inds]
            labels_unmapped = labels
            # print("labels after unmap:", labels_unmapped.unique())
            # print("labels at pos_inds_unmapped:", labels_unmapped[inside_inds[pos_inds]])

            # print("labels at pos_inds after unmap:", labels[pos_inds_unmapped])
            # print("bbox_gt at pos_inds after unmap:", bbox_gt[pos_inds_unmapped])

            # print("== After unmap ==")
            # print(f"labels shape: {labels.shape}, unique labels: {torch.unique(labels)}")
            # print(f"label_weights shape: {label_weights.shape}, sum: {label_weights.sum()}")
            # print(f"bbox_gt shape: {bbox_gt.shape}")
            # print(f"bbox_gt sample (first 5): {bbox_gt[:5]}")
            # print(f"pos_proposals shape: {pos_proposals.shape}")
            # print(f"proposals_weights shape: {proposals_weights.shape}, sum: {proposals_weights.sum()}")

        # print(f"flat_proposals shape: {flat_proposals.shape}")
        # print(f"valid_flags shape: {valid_flags.shape}")
        # print("inside_flags dtype:", inside_flags.dtype, "sum:", inside_flags.sum())
        # print("inside_flags unique:", torch.unique(inside_flags))
        # print(f"proposals shape: {proposals.shape}")
        # print(f"assigner type: {type(assigner)}")

        return (labels, label_weights, bbox_gt, pos_proposals,
            proposals_weights, pos_inds_unmapped, neg_inds_unmapped, sampling_result)

    def get_targets(self,
                proposals_list: List[List[Tensor]],
                valid_flag_list: List[List[Tensor]],
                batch_gt_instances: InstanceList,
                batch_img_metas: List[dict],
                batch_gt_instances_ignore: OptInstanceList = None,
                stage: str = 'init',
                unmap_outputs: bool = True,
                return_sampling_results: bool = False,
                num_level_proposals: Optional[List[int]] = None) -> tuple:

        # Convert gt_instances to HorizontalBoxes
        for gt_inst in batch_gt_instances:
            if hasattr(gt_inst, 'bboxes'):
                if not isinstance(gt_inst.bboxes, HorizontalBoxes):
                    gt_inst.bboxes = HorizontalBoxes(gt_inst.bboxes if not hasattr(gt_inst.bboxes, 'tensor') 
                                                else gt_inst.bboxes.tensor)

        # For init stage, convert points to small boxes
        if stage == 'init':
            new_proposals_list = []
            for points in proposals_list:
                img_proposals = []
                for lvl_points in points:
                    if lvl_points.size(-1) > 2:
                        lvl_points = lvl_points[..., :2]
                    scale = self.point_base_scale
                    boxes = torch.cat([
                        lvl_points - scale / 2,
                        lvl_points + scale / 2
                    ], dim=-1)
                    img_proposals.append(boxes)
                new_proposals_list.append(img_proposals)
            proposals_list = new_proposals_list

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * len(batch_img_metas)

        # --- FIX START ---
        # Flatten each image's proposals from list of levels into single tensor
        # def concat_levels(img_level_list):
        #     return [torch.cat(levels, dim=0) if isinstance(levels, (list, tuple)) else levels for levels in img_level_list]
        def concat_levels(img_level_list):
            if isinstance(img_level_list, (list, tuple)):
                return torch.cat(img_level_list, dim=0)
            return img_level_list  # Already a tensor, no need to concat

        flat_proposals_list = [concat_levels(p) for p in proposals_list]
        flat_valid_flag_list = [concat_levels(v) for v in valid_flag_list]

        # --- FIX END ---

        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
        all_proposal_weights, pos_inds_list, neg_inds_list,
        sampling_results_list) = multi_apply(
            self._get_targets_single,
            flat_proposals_list,             # flattened proposals
            flat_valid_flag_list,            # flattened valid flags
            batch_gt_instances,
            batch_gt_instances_ignore,
            stage=stage,
            unmap_outputs=unmap_outputs)
        
        # print("Assigned GT counts per image:", [p.sum().item() for p in all_proposal_weights])
        # print("Nonzero bbox targets per image:", [torch.count_nonzero(b).item() for b in all_bbox_gt])

        # Ensure num_level_proposals is known
        if num_level_proposals is None:
            if isinstance(proposals_list[0], (list, tuple)):
                num_level_proposals = [p.size(0) for p in proposals_list[0]]
            else:
                num_level_proposals = [proposals_list[0].size(0)]

        # --- FIX START ---
        # Flatten all outputs per image before calling images_to_levels
        def flatten_preds(per_img_list):
            return [torch.cat(lvl_preds if isinstance(lvl_preds, (list, tuple)) else [lvl_preds]) for lvl_preds in per_img_list]

        all_labels = flatten_preds(all_labels)
        all_label_weights = flatten_preds(all_label_weights)
        all_bbox_gt = flatten_preds(all_bbox_gt)
        all_proposals = flatten_preds(all_proposals)
        all_proposal_weights = flatten_preds(all_proposal_weights)
        # --- FIX END ---

        # Convert per-image to per-level
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights, num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        bbox_weights_list = images_to_levels(all_proposal_weights, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)

        avg_factor_init = sum([res.avg_factor for res in sampling_results_list])
        avg_factor_refine = avg_factor_init

        return (labels_list, label_weights_list, 
                bbox_gt_list, bbox_weights_list,
                proposals_list, all_proposal_weights,
                avg_factor_init, avg_factor_refine,
                pos_inds_list, neg_inds_list)

    def loss_by_feat_single(self, cls_score, pts_pred_init, pts_pred_refine,
                        bbox_pred_init, bbox_pred_refine, labels, label_weights,
                        bbox_gt_init, bbox_weights_init, bbox_gt_refine,
                        bbox_weights_refine, avg_factor_init, avg_factor_refine,
                        stride):  # <-- Add stride here
        # Classification loss
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        valid_mask = labels >= 0
        if not valid_mask.any():
            return cls_score.sum() * 0, pts_pred_init.sum() * 0, pts_pred_refine.sum() * 0

        loss_cls = self.loss_cls(
            cls_score[valid_mask],
            labels[valid_mask],
            label_weights[valid_mask],
            avg_factor=max(1, valid_mask.sum()))

        # Regression losses
        pts_pred_init = pts_pred_init.permute(0, 2, 3, 1).reshape(-1, 2 * self.num_points)
        pts_pred_refine = pts_pred_refine.permute(0, 2, 3, 1).reshape(-1, 2 * self.num_points)

        # 🔥 Multiply by stride to scale to image space
        pts_pred_init = pts_pred_init * stride # /4.0 # working
        pts_pred_refine = pts_pred_refine * stride # /4.0 # working

        bbox_pred_init = self.points2bbox(pts_pred_init, y_first=False)
        bbox_pred_refine = self.points2bbox(pts_pred_refine, y_first=False)

        assert bbox_pred_init.shape[0] == labels.shape[0], \
            f"Mismatch: bbox_pred_init={bbox_pred_init.shape}, labels={labels.shape}"

        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)

        pos_inds = (labels >= 0) & (labels < self.num_classes)
        num_pos = pos_inds.sum().item()

        if num_pos > 0:
            pos_inds = pos_inds.nonzero(as_tuple=False).squeeze(1)

            loss_pts_init = self.loss_bbox_init(
                bbox_pred_init[pos_inds],
                bbox_gt_init[pos_inds],
                bbox_weights_init[pos_inds],
                avg_factor=num_pos)

            loss_pts_refine = self.loss_bbox_refine(
                bbox_pred_refine[pos_inds],
                bbox_gt_refine[pos_inds],
                bbox_weights_refine[pos_inds],
                avg_factor=num_pos)
        else:
            loss_pts_init = bbox_pred_init.sum() * 0
            loss_pts_refine = bbox_pred_refine.sum() * 0

        return loss_cls, loss_pts_init, loss_pts_refine

    def get_points(self, featmap_sizes, batch_input_shape=None, device=None, batch_size=1):
        points_list = []
        valid_flag_list = []

        for i, featmap_size in enumerate(featmap_sizes):
            h, w = featmap_size
            stride = self.strides[i]

            shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            points = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=-1) + stride // 2  # [N_lvl, 2]

            if batch_input_shape is not None:
                img_h, img_w = batch_input_shape
                valid_x = (shift_x.reshape(-1) >= 0) & (shift_x.reshape(-1) < img_w)
                valid_y = (shift_y.reshape(-1) >= 0) & (shift_y.reshape(-1) < img_h)
                valid = valid_x & valid_y  # [N_lvl]
            else:
                valid = torch.ones(points.size(0), dtype=torch.bool, device=device)

            # Add batch dimension by repeating for batch_size
            points_batch = points.unsqueeze(0).repeat(batch_size, 1, 1)          # [B, N_lvl, 2]
            valid_batch = valid.unsqueeze(0).repeat(batch_size, 1)               # [B, N_lvl]

            points_list.append(points_batch)
            valid_flag_list.append(valid_batch)

        return points_list, valid_flag_list
    
    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        pts_preds_init: List[Tensor],
        pts_preds_refine: List[Tensor],
        bbox_preds_init: List[Tensor],
        bbox_preds_refine: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        batch_size = len(batch_img_metas)
        
        # Get points for each level
        points_list, valid_flag_list = self.get_points(
            featmap_sizes, batch_img_metas[0]['batch_input_shape'], device, batch_size)
        
        (labels_list, label_weights_list, 
        bbox_gt_init_list, bbox_weights_init_list,
        bbox_gt_refine_list, bbox_weights_refine_list,
        avg_factor_init, avg_factor_refine,
        pos_inds_list, neg_inds_list) = self.get_targets(
            points_list, valid_flag_list, batch_gt_instances, batch_img_metas,
            batch_gt_instances_ignore)
        
        strides = [self.point_strides[i] for i in range(len(cls_scores))]
        # Compute losses for each level separately
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            pts_preds_init,
            pts_preds_refine,
            bbox_preds_init,
            bbox_preds_refine,
            labels_list,
            label_weights_list,
            bbox_gt_init_list,
            bbox_weights_init_list,
            bbox_gt_refine_list,
            bbox_weights_refine_list,
            [avg_factor_init] * len(cls_scores),
            [avg_factor_refine] * len(cls_scores),
            strides
        )
        
        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine
        }
        return loss_dict_all
    
    def get_proposals(self, cls_scores, bbox_preds, score_factors, img_metas, cfg):
        assert img_metas is not None, "img_metas must be provided"

        device = cls_scores[0].device
        batch_size = cls_scores[0].size(0)
        num_levels = len(cls_scores)

        proposals_list = [[] for _ in range(batch_size)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]  # (N,1,H,W) or (N, H*W)
            bbox_pred = bbox_preds[lvl]  # (N,4,H,W) or (N, H*W, 4)

            # Handle cls_score shape
            if cls_score.dim() == 4:
                N, _, H, W = cls_score.shape
                cls_score = cls_score.permute(0, 2, 3, 1).reshape(N, -1)  # (N, H*W)
            elif cls_score.dim() == 2:
                N = cls_score.size(0)  # Already flattened
            else:
                raise ValueError(f"Unexpected cls_score shape: {cls_score.shape}")

            # Handle bbox_pred shape
            if bbox_pred.dim() == 4:
                N2, _, H, W = bbox_pred.shape
                assert N2 == N, f"Inconsistent batch size: {N} vs {N2}"
                C = bbox_pred.size(1)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(N, -1, C)

                # print(f"bbox_pred type: {type(bbox_pred)}, bbox_pred shape: {getattr(bbox_pred, 'shape', None)}")
                # print(f"bbox_pred data: {bbox_pred}")

            elif bbox_pred.dim() == 3:
                N2 = bbox_pred.size(0)
                assert N2 == N, f"Inconsistent batch size: {N} vs {N2}"
                # Already (N, H*W, 4)
            else:
                raise ValueError(f"Unexpected bbox_pred shape: {bbox_pred.shape}")

            for img_id in range(N):
                scores = cls_score[img_id]  # (H*W,)
                bboxes = bbox_pred[img_id]  # (H*W, 4)

                score_thresh = getattr(cfg, 'score_thr', 0.05)
                keep_idxs = scores > score_thresh
                scores = scores[keep_idxs]
                bboxes = bboxes[keep_idxs]

                if scores.numel() == 0:
                    continue

                img_meta = img_metas[img_id]
                img_shape = img_meta['img_shape'][:2]

                num_proposals = bboxes.size(0)
                decoded_bboxes_list = []
                for i in range(num_proposals):
                    points = bboxes[i].view(-1, 2)  # reshape to (18, 2) if 36 = 18 points * 2 coords
                    x1 = points[:, 0].clamp(min=0, max=img_shape[1]).min()
                    y1 = points[:, 1].clamp(min=0, max=img_shape[0]).min()
                    x2 = points[:, 0].clamp(min=0, max=img_shape[1]).max()
                    y2 = points[:, 1].clamp(min=0, max=img_shape[0]).max()
                    decoded_bboxes_list.append(torch.tensor([x1, y1, x2, y2], device=points.device))

                decoded_bboxes = torch.stack(decoded_bboxes_list)  # (num_proposals, 4)
                proposals = torch.cat([decoded_bboxes, scores.unsqueeze(1)], dim=1)  # (num_proposals, 5)

                nms_pre = getattr(cfg, 'nms_pre', 1000)
                if proposals.size(0) > nms_pre:
                    scores_for_sort = proposals[:, 4]
                    _, topk_idxs = scores_for_sort.topk(nms_pre)
                    proposals = proposals[topk_idxs]

                nms_thresh = getattr(cfg, 'nms_thresh', 0.7)
                keep = nms(proposals[:, :4], proposals[:, 4], nms_thresh)
                proposals = proposals[keep]

                max_per_img = getattr(cfg, 'max_per_img', 1000)
                if proposals.size(0) > max_per_img:
                    proposals = proposals[:max_per_img]

                proposals_list[img_id] = [proposals]
                #proposals_list[img_id].append(proposals)

        final_proposals = []
        for img_id in range(batch_size):
            if len(proposals_list[img_id]) == 0:
                final_proposals.append(torch.empty((0, 5), device=device))
            else:
                proposals_per_img = torch.cat(proposals_list[img_id], dim=0)

                max_per_img = getattr(cfg, 'max_per_img', 1000)
                if proposals_per_img.size(0) > max_per_img:
                    scores_for_sort = proposals_per_img[:, 4]
                    _, topk_idxs = scores_for_sort.topk(max_per_img)
                    proposals_per_img = proposals_per_img[topk_idxs]

                final_proposals.append(proposals_per_img)

        return final_proposals

    def loss_and_predict(self,
                    x,
                    batch_data_samples,
                    proposal_cfg=None):
    
        cls_scores, pts_preds_init, pts_preds_refine, bbox_preds_init, bbox_preds_refine = self.predict_raw(x)

        batch_gt_instances = [sample.gt_instances for sample in batch_data_samples]
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]

        losses = self.loss_by_feat(
            cls_scores,
            pts_preds_init,
            pts_preds_refine,
            bbox_preds_init,
            bbox_preds_refine,
            batch_gt_instances,
            batch_img_metas
        )

        # print("\n=== Loss breakdown ===")
        # for k, v in losses.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {v.item():.6f}")
        #     else:
        #         print(f"{k}: {v}")

        if proposal_cfg is None:
            raise ValueError("proposal_cfg must be provided")

        # pass batch_img_metas explicitly here as positional argument
        proposal_list = self.get_proposals(
            cls_scores,
            bbox_preds_refine,
            score_factors=None,
            img_metas=batch_img_metas,
            cfg=proposal_cfg
        )

        proposal_list = [InstanceData(bboxes=p) for p in proposal_list]

        return losses, proposal_list
    
    def predict(self, x, batch_img_metas, rescale=False):
        # x: list of feature maps
        # batch_img_metas: list of dicts (image info)

        cls_scores, pts_preds_init, pts_preds_refine, bbox_preds_init, bbox_preds_refine = self.predict_raw(x)

        results = self.predict_by_feat(
            cls_scores=cls_scores,
            pts_preds_refine=pts_preds_refine,
            bbox_preds_init=bbox_preds_init,
            batch_img_metas=batch_img_metas,
            cfg=None,
            rescale=rescale,
            with_nms=True
        )
        return results

    def predict_by_feat(self,
                   cls_scores: List[Tensor],
                   pts_preds_refine: List[Tensor],
                   bbox_preds_init: List[Tensor],
                   batch_img_metas: List[dict],
                   cfg: Optional[ConfigDict] = None,
                   rescale: bool = False,
                   with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features into bbox results.

        Args:
            cls_scores (list[Tensor]): Classification scores for each scale level.
            pts_preds_refine (list[Tensor]): Refined points predictions for each
                scale level, each is a 4D-tensor, the channel number is
                num_points * 2.
            bbox_preds_init (list[Tensor]): Initial bbox predictions for each scale
                level, each is a 4D-tensor, the channel number is num_points * 2.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (ConfigDict, optional): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
            with_nms (bool): If True, do nms before return boxes.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys:
                - scores (Tensor): Classification scores, has a shape
                (num_instance,)
                - labels (Tensor): Labels of bboxes, has a shape
                (num_instance,)
                - bboxes (Tensor): Has a shape (num_instance, 4),
                the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(pts_preds_refine)
        
        # Get test config
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        
        # Convert points to priors (anchor points)
        mlvl_priors = []
        for i in range(num_levels):
            stride = self.point_strides[i]
            feat_h, feat_w = cls_scores[i].shape[-2:]
            h_range = torch.arange(
                0, feat_h, device=device, dtype=torch.float32) * stride + stride // 2
            w_range = torch.arange(
                0, feat_w, device=device, dtype=torch.float32) * stride + stride // 2
            y, x = torch.meshgrid(h_range, w_range, indexing='ij')
            prior = torch.stack([x, y], dim=-1).reshape(-1, 2)
            mlvl_priors.append(prior)
        
        # Process each image in the batch
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            #img_shape = img_meta['img_shape']

            # Get single image predictions from all levels
            cls_score_list = [
                cls_scores[i][img_id].unsqueeze(0) for i in range(num_levels)
            ]
            bbox_pred_list = [
                pts_preds_refine[i][img_id].unsqueeze(0) for i in range(num_levels)
            ]
            
            # Process predictions for single image
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=None,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            
            img_id = img_meta['metainfo']['img_id'] if 'metainfo' in img_meta and 'img_id' in img_meta['metainfo'] else -1
            result_list.append(results) 
        
        return result_list

    # Same as base_dense_head/_get_bboxes_single except self._bbox_decode
    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list)
        img_shape = img_meta.metainfo['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list, mlvl_priors)):

            # If batch dim exists, remove it (assuming batch=1)
            if cls_score.dim() == 4:
                cls_score = cls_score.squeeze(0)  # (C, H, W)
            if bbox_pred.dim() == 4:
                bbox_pred = bbox_pred.squeeze(0)  # (C, H, W)
            
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            C, H, W = bbox_pred.shape
            
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, C)

            cls_score = cls_score.permute(1, 2,0).reshape(-1, self.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            min_len = min(bbox_pred.shape[0], priors.shape[0])
            bbox_pred = bbox_pred[:min_len]
            priors = priors[:min_len]

            print("bbox_pred shape:", bbox_pred.shape)
            print("priors shape:", priors.shape)

            assert bbox_pred.shape[0] == priors.shape[0], \
                f"Shape mismatch after filtering: bbox_pred {bbox_pred.shape}, priors {priors.shape}"

            bboxes = self._bbox_decode(priors, bbox_pred,
                                    self.point_strides[level_idx],
                                    img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = HorizontalBoxes.cat(mlvl_bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def _bbox_decode(self, points: Tensor, bbox_pred: Tensor, stride: int,
                max_shape: Tuple[int, int]) -> Tensor:
        """
        Decode 18-channel bbox prediction into horizontal bounding boxes.

        Args:
            points (Tensor): Anchor points of shape (N, 2).
            bbox_pred (Tensor): Predicted offsets of shape (N, 18).
            stride (int): Stride of the feature map.
            max_shape (tuple): Image shape (height, width).

        Returns:
            HorizontalBoxes: Decoded bounding boxes of shape (N, 4).
        """
        assert bbox_pred.shape[1] == 18, f"Expected 18 channels but got {bbox_pred.shape[1]}"

        # Reshape to (N, 9, 2)
        pred_offsets = bbox_pred.view(-1, 9, 2)  # (N, 9, 2)

        # Compute absolute point coordinates
        x_centers = points[:, 0].unsqueeze(1)  # (N, 1)
        y_centers = points[:, 1].unsqueeze(1)  # (N, 1)

        x_offsets = pred_offsets[:, :, 0] * stride  # (N, 9)
        y_offsets = pred_offsets[:, :, 1] * stride  # (N, 9)

        x_coords = x_centers + x_offsets  # (N, 9)
        y_coords = y_centers + y_offsets  # (N, 9)

        # Compute axis-aligned bbox that encloses all points
        x_min = x_coords.min(dim=1).values.clamp(0, max_shape[1])
        y_min = y_coords.min(dim=1).values.clamp(0, max_shape[0])
        x_max = x_coords.max(dim=1).values.clamp(0, max_shape[1])
        y_max = y_coords.max(dim=1).values.clamp(0, max_shape[0])

        decoded_bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        return HorizontalBoxes(decoded_bboxes)
