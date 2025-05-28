# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

@TASK_UTILS.register_module()
class PointAssigner(BaseAssigner):
    def __init__(self, scale: int = 4, pos_num: int = 3, radius: float = 1.5) -> None:
        self.scale = scale
        self.pos_num = pos_num
        self.radius = radius

    def assign(self, pred_instances, gt_instances, gt_instances_ignore=None, **kwargs):
        points = pred_instances.priors
        points_xy = points[:, :2]
        points_stride = points[:, 2] if points.size(1) > 2 else points.new_ones(points.size(0)) * self.scale
        
        gt_bboxes = gt_instances.bboxes.tensor
        assigned_gt_inds = points.new_full((len(points),), 0, dtype=torch.long)
        max_overlaps = points.new_zeros(len(points))
        labels = points.new_full((len(points),), 0, dtype=torch.long)  # Initialize labels
        
        if len(gt_bboxes) == 0:
            return AssignResult(
                num_gts=0,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=labels)  # Return initialized labels

        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        dist_matrix = torch.cdist(points_xy, gt_centers)
        
        for gt_idx in range(len(gt_bboxes)):
            radius = self.radius * points_stride
            within_radius = dist_matrix[:, gt_idx] <= radius
            if within_radius.any():
                assigned_gt_inds[within_radius] = gt_idx + 1
                max_overlaps[within_radius] = 1.0 - (dist_matrix[within_radius, gt_idx] / (radius[within_radius] + 1e-6))
                labels[within_radius] = 1  # Set positive labels
                
                if within_radius.sum() < self.pos_num:
                    _, topk_idx = torch.topk(-dist_matrix[:, gt_idx], k=self.pos_num)
                    assigned_gt_inds[topk_idx] = gt_idx + 1
                    max_overlaps[topk_idx] = 1.0
                    labels[topk_idx] = 1

        return AssignResult(
            num_gts=len(gt_bboxes),
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=labels)  # Return the labels
