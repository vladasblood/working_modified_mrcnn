from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
import torch

@MODELS.register_module()
class RepPointsDataPreprocessor(DetDataPreprocessor):
    def forward(self, data, training=False):
        # Standard preprocessing
        data = super().forward(data, training)
        
        if not training:
            return data

        inputs, data_samples = data['inputs'], data['data_samples']
        
        for data_sample in data_samples:
            if not hasattr(data_sample, 'gt_instances'):
                continue

            bboxes = data_sample.gt_instances.bboxes
            # Safely get tensor in case HorizontalBoxes or Tensor
            if hasattr(bboxes, 'tensor'):
                bboxes_tensor = bboxes.tensor
            else:
                bboxes_tensor = bboxes

            points = self._bbox_to_points(bboxes_tensor)

            # Ensure we retain all fields in gt_instances
            data_sample.gt_instances.points = points
            data_sample.gt_instances.bboxes = bboxes  # Retain original boxes

        return {'inputs': inputs, 'data_samples': data_samples}

    def _bbox_to_points(self, bboxes):
        """Convert bboxes to point format expected by RepPoints"""
        if not isinstance(bboxes, torch.Tensor) or bboxes.numel() == 0:
            return torch.empty((0, 9, 2), device=bboxes.device)

        # Get top-left and bottom-right
        x1, y1, x2, y2 = bboxes.unbind(-1)
        width = x2 - x1
        height = y2 - y1

        # Grid: 3x3 layout in normalized bbox space
        grid = torch.tensor([
            [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
            [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
            [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
        ], device=bboxes.device)

        # (N, 9, 2) = (N, 1, 2) + (1, 9, 2) * (N, 1, 2)
        points = torch.stack([x1, y1], dim=1)[:, None, :] + grid[None, :, :] * torch.stack([width, height], dim=1)[:, None, :]

        return points
