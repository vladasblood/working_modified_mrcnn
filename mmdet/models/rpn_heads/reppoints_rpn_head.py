def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Args:
        func (callable): function to be applied.
        *args (list): multiple list of arguments.
        **kwargs (dict): keyword arguments.

    Returns:
        tuple: a tuple of list of results.
    """
    pfunc = lambda arg: func(*arg, **kwargs) if isinstance(arg, (tuple, list)) else func(arg, **kwargs)
    map_results = map(pfunc, zip(*args))
    return tuple(map(list, zip(*map_results)))

import torch

from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, scale_boxes, empty_box_as
from mmcv.ops import batched_nms
from mmdet.models.dense_heads.reppoints_head import RepPointsHead

@MODELS.register_module()
class RepPointsRPNHead(RepPointsHead):
    """Wrapper of RepPoints to act as RPNHead in two-stage detectors."""

    def __init__(self, *args, **kwargs):
        kwargs['num_classes'] = 1  # RPN is class-agnostic
        super().__init__(*args, **kwargs)

    def predict_raw(self, x):
        return multi_apply(self.forward_single, x)

    def simple_test_rpn(self, x, img_metas):
        """Inference: get proposals compatible with RoI heads."""
        cls_scores, bbox_preds, score_factors = self.predict_raw(x)
        return self.get_proposals(cls_scores, bbox_preds, score_factors, img_metas, self.test_cfg)

    def get_proposals(self, cls_scores, bbox_preds, score_factors, img_metas, cfg):
        """Generate proposal boxes per image."""
        result_list = self.get_bboxes(
            cls_scores, bbox_preds, score_factors, img_metas, cfg)

        final_results = []
        for proposals, meta in zip(result_list, img_metas):
            results = InstanceData()
            if proposals.numel() == 0:
                results.bboxes = empty_box_as(proposals)
                results.scores = torch.zeros((0,), device=proposals.device)
                results.labels = torch.zeros((0,), dtype=torch.long, device=proposals.device)
            else:
                if cfg.get('rescale', False):
                    scale_factor = [1 / s for s in meta['scale_factor']]
                    proposals[:, :4] = scale_boxes(proposals[:, :4], scale_factor)
                scores = proposals[:, -1]
                bboxes = proposals[:, :4]

                # optional: batched NMS here if needed
                det_bboxes, keep = batched_nms(bboxes, scores, torch.zeros_like(scores), cfg.nms)
                det_bboxes = det_bboxes[:cfg.max_per_img]

                results.bboxes = det_bboxes[:, :4]
                results.scores = det_bboxes[:, -1]
                results.labels = torch.zeros_like(results.scores, dtype=torch.long)

            final_results.append(results)
        return final_results

    def loss_and_proposals(self, x, batch_data_samples, proposal_cfg):
        """Forward + loss + generate proposals for training."""
        losses = self.loss(x, batch_data_samples)
        if proposal_cfg is not None:
            cls_scores, bbox_preds, score_factors = self.predict_raw(x)
            proposal_list = self.get_proposals(
                cls_scores, bbox_preds, score_factors,
                [sample.metainfo for sample in batch_data_samples],
                proposal_cfg)
        else:
            proposal_list = None
        return dict(
            loss_rpn_cls=losses['loss_cls'],
            loss_rpn_bbox=losses['loss_bbox']
        ), proposal_list
