import torch

from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, scale_boxes, empty_box_as
from mmcv.ops import batched_nms
from mmdet.models.dense_heads.reppoints_head import RepPointsHead
from mmdet.structures import DetDataSample
from torchvision.ops import nms

@MODELS.register_module()
class RepPointsRPNHead(RepPointsHead):
    """Wrapper of RepPoints to act as RPNHead in two-stage detectors."""

    def __init__(self, *args, **kwargs):
        kwargs['num_classes'] = 1  # RPN is class-agnostic
        super().__init__(*args, **kwargs)
    
    def loss_and_proposals(self, x, batch_data_samples, proposal_cfg):

        losses, proposal_list = self.loss_and_predict(
            x,
            batch_data_samples,
            proposal_cfg=proposal_cfg
        )  

        return dict(
            loss_cls=losses['rpn_loss_cls'],
            loss_pts_init=losses['rpn_loss_pts_init'],
            loss_pts_refine=losses['rpn_loss_pts_refine'],
        ), proposal_list