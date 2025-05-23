# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class PointAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __init__(self, scale: int = 4, pos_num: int = 3) -> None:
        self.scale = scale
        self.pos_num = pos_num

    def assign(self,
           pred_instances: InstanceData,
           gt_instances: InstanceData,
           gt_instances_ignore: Optional[InstanceData] = None,
           **kwargs) -> AssignResult:
        """Assign gt to points.

        (docstring unchanged for brevity)
        """
    # Extract gt_bboxes safely from gt_instances
        if gt_instances is None or not hasattr(gt_instances, 'bboxes'):
            # No gt instances or no bboxes attribute -> no ground truth
            gt_bboxes = None
            gt_labels = None
        else:
            gt_bboxes = gt_instances.bboxes
            gt_labels = gt_instances.labels if hasattr(gt_instances, 'labels') else None

        # Unwrap gt_bboxes if wrapped in InstanceData (rare but possible)
        if hasattr(gt_bboxes, 'bboxes'):
            gt_bboxes = gt_bboxes.bboxes

        # Check if gt_bboxes is valid tensor and not empty
        if (gt_bboxes is None
            or not isinstance(gt_bboxes, torch.Tensor)
            or gt_bboxes.numel() == 0
            or gt_bboxes.shape[0] == 0):
            # No ground truth boxes, assign all points as background
            num_points = pred_instances.priors.shape[0]
            assigned_gt_inds = pred_instances.priors.new_full((num_points,), 0, dtype=torch.long)
            assigned_labels = pred_instances.priors.new_full((num_points,), -1, dtype=torch.long)
            
            # if (assigned_gt_inds > 0).sum() == 0:
            #   print("[WARNING] No points assigned in init stage!")

            # print("[DEBUG] Returning AssignResult (no GTs):")
            # print(f"  assigned_gt_inds: {assigned_gt_inds}")
            # print(f"  max_overlaps: None")
            # print(f"  labels: {assigned_labels}")

            dummy_overlaps = pred_instances.priors.new_zeros((num_points,))

            return AssignResult(
                num_gts=0,
                gt_inds=assigned_gt_inds,
                max_overlaps=dummy_overlaps,
                labels=assigned_labels)

        # Now it's safe to proceed
        points = pred_instances.priors
        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        num_points = points.shape[0]
        num_gts = gt_bboxes.shape[0]

        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
        scale = self.scale
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                        torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        assigned_gt_inds = points.new_zeros((num_points,), dtype=torch.long)
        assigned_gt_dist = points.new_full((num_points,), float('inf'))
        points_range = torch.arange(num_points, device=points.device)

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]

            # print(f'[DEBUG] GT #{idx}: assigned level={gt_lvl},'
            # f'matched point candidates={len(points_index)}')

            lvl_points = points_xy[lvl_idx, :]
            gt_point = gt_bboxes_xy[[idx], :]
            gt_wh = gt_bboxes_wh[[idx], :]
            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
            min_dist, min_dist_index = torch.topk(points_gt_dist, self.pos_num, largest=False)
            min_dist_points_index = points_index[min_dist_index]
            less_than_recorded_index = min_dist < assigned_gt_dist[min_dist_points_index]
            min_dist_points_index = min_dist_points_index[less_than_recorded_index]
            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[less_than_recorded_index]

        # print(f'[DEBUG] Total assigned positive points: {(assigned_gt_inds > 0).sum().item()}')

        assigned_labels = assigned_gt_inds.new_full((num_points,), -1)
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0 and gt_labels is not None:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        
        # print("[DEBUG] Returning AssignResult (with GTs):")
        # print(f"  num_gts: {num_gts}")
        # print(f"  assigned_gt_inds: {assigned_gt_inds}")
        # print(f"  max_overlaps: None")
        # print(f"  labels: {assigned_labels}")

        dummy_overlaps = pred_instances.priors.new_zeros((num_points,))

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=dummy_overlaps,
            labels=assigned_labels)