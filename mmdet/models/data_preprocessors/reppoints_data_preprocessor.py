from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures.bbox import HorizontalBoxes
import torch

@MODELS.register_module()
class RepPointsDataPreprocessor(DetDataPreprocessor):
    def forward(self, data, training=False):
        # First do standard preprocessing
        data = super().forward(data, training)
        
        if not training:
            return data
            
        # Handle the data structure correctly
        inputs, data_samples = data['inputs'], data['data_samples']
        
        for i, data_sample in enumerate(data_samples):
            if not hasattr(data_sample, 'gt_instances'):
                continue
                
            # Get bboxes (keeping them in HorizontalBoxes format)
            bboxes = data_sample.gt_instances.bboxes
            points = self._bbox_to_points(bboxes.tensor if hasattr(bboxes, 'tensor') else bboxes)
            
            # Create new InstanceData with points
            new_instances = InstanceData()
            for k, v in data_sample.gt_instances.items():
                new_instances[k] = v
            new_instances.points = points
            data_sample.gt_instances = new_instances
        
        return {'inputs': inputs, 'data_samples': data_samples}

    def _bbox_to_points(self, bboxes):
        """Convert bboxes to point format expected by RepPoints"""
        if not isinstance(bboxes, torch.Tensor) or len(bboxes) == 0:
            return None
            
        # Convert to center points (x,y)
        centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
        
        # Create 9 points per box (3x3 grid)
        x1, y1, x2, y2 = bboxes.unbind(-1)
        width = x2 - x1
        height = y2 - y1
        
        # Generate relative offsets for 9 points
        grid = torch.tensor([
            [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
            [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
            [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
        ], device=bboxes.device)
        
        # Scale offsets to each bbox
        points = bboxes[:, None, :2] + grid[None, :, :] * torch.stack([width, height], -1)[:, None, :]
        
        return points
