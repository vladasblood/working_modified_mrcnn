# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

@TASK_UTILS.register_module()
class PointAssigner(BaseAssigner):
    def __init__(self, scale: int = 4, pos_num: int = 3) -> None:
        self.scale = scale
        self.pos_num = pos_num

    def assign(self, pred_instances, gt_instances, gt_instances_ignore=None, **kwargs):
        # priors: [N, 3] -> (x, y, stride)
        points = pred_instances.priors
        points_xy = points[:, :2]  # Use point coordinates directly

        gt_bboxes = gt_instances.bboxes.tensor
        assigned_gt_inds = torch.zeros(len(points), dtype=torch.long, device=points.device)
        max_overlaps = torch.zeros(len(points), dtype=torch.float, device=points.device)

        if len(gt_bboxes) == 0:
            return AssignResult(
                num_gts=0,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_gt_inds.new_full((len(points),), -1))

        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        dist_matrix = torch.cdist(points_xy, gt_centers)

        for gt_idx in range(len(gt_bboxes)):
            _, topk_idx = torch.topk(-dist_matrix[:, gt_idx], k=self.pos_num)
            assigned_gt_inds[topk_idx] = gt_idx + 1
            max_overlaps[topk_idx] = 1.0 / (1.0 + dist_matrix[topk_idx, gt_idx])

        num_pos = (assigned_gt_inds > 0).sum().item()
        print(f"[PointAssigner] Assigned {num_pos} positive points out of {len(points)}")

        return AssignResult(
            num_gts=len(gt_bboxes),
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_gt_inds.new_full((len(points),), -1))
