def multi_apply(func, *args, **kwargs):
    pfunc = lambda arg: func(*arg, **kwargs) if isinstance(arg, (tuple, list)) else func(arg, **kwargs)
    map_results = map(pfunc, zip(*args))
    results = tuple(map(list, zip(*map_results)))
    
    print("[multi_apply] Outputs:")
    for i, res in enumerate(results):
        print(f" - Result {i}: {[r.shape if isinstance(r, torch.Tensor) else type(r) for r in res]}")
    
    return results

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

    def predict_raw(self, x):
        out = multi_apply(self.forward_single, x)
        print("[predict_raw] Returned cls_scores, bbox_preds, score_factors.")
        for i, o in enumerate(out):
            print(f"  - Output {i}: {[x.shape if isinstance(x, torch.Tensor) else type(x) for x in o]}")
        return out

    def simple_test_rpn(self, x, img_metas):
        """Inference: get proposals compatible with RoI heads."""
        cls_scores, bbox_preds, score_factors = self.predict_raw(x)
        return self.get_proposals(cls_scores, bbox_preds, score_factors, img_metas, self.test_cfg)

    def get_proposals(self, cls_scores, bbox_preds, score_factors, img_metas, cfg):
        """Generate proposal boxes per image."""
        assert img_metas is not None, "img_metas must be provided"

        device = cls_scores[0].device
        batch_size = cls_scores[0].size(0)
        num_levels = len(cls_scores)

        proposals_list = [[] for _ in range(batch_size)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]  # (N,1,H,W)
            bbox_pred = bbox_preds[lvl]  # (N,36,H,W)

            print(f"Level {lvl} bbox_pred type and shape:", type(bbox_preds[lvl]), getattr(bbox_preds[lvl], 'shape', None))
            print(f"Level {lvl} cls_score type and shape:", type(cls_scores[lvl]), getattr(cls_scores[lvl], 'shape', None))

            N, C_cls, H, W = cls_score.shape
            N, C_bbox, H, W = bbox_pred.shape

            assert C_bbox == 36, "Expect 36 channels for RepPoints bbox_pred"

            cls_score = cls_score.permute(0, 2, 3, 1).reshape(N, -1)  # (N, H*W)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(N, -1, C_bbox)

            for img_id in range(N):
                scores = cls_score[img_id]  # (H*W,)
                bboxes = bbox_pred[img_id]  # (H*W, 4 or 36)

                score_thresh = getattr(cfg, 'score_thr', 0.05)
                keep_idxs = scores > score_thresh
                scores = scores[keep_idxs]
                bboxes = bboxes[keep_idxs]

                if scores.numel() == 0:
                    continue

                img_meta = img_metas[img_id]
                img_shape = img_meta['img_shape'][:2]

                decoded_bboxes = bboxes

                # Safe non-inplace clamping
                x_coords = decoded_bboxes[:, 0::2].clamp(min=0, max=img_shape[1])
                y_coords = decoded_bboxes[:, 1::2].clamp(min=0, max=img_shape[0])
                decoded_bboxes = decoded_bboxes.clone()
                decoded_bboxes[:, 0::2] = x_coords
                decoded_bboxes[:, 1::2] = y_coords

                # Filter out boxes with non-positive width or height
                x1, y1, x2, y2 = decoded_bboxes[:, 0], decoded_bboxes[:, 1], decoded_bboxes[:, 2], decoded_bboxes[:, 3]
                valid_mask = (x2 > x1) & (y2 > y1)
                decoded_bboxes = decoded_bboxes[valid_mask]
                scores = scores[valid_mask]

                if scores.numel() == 0:
                    continue

                proposals = torch.cat([decoded_bboxes, scores.unsqueeze(1)], dim=1)  # (num,5)

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

                proposals_list[img_id].append(proposals)

        final_proposals = []
        for img_id in range(batch_size):
            if len(proposals_list[img_id]) == 0:
                empty_bboxes = torch.empty((0, 4), device=device)
                empty_scores = torch.empty((0,), device=device)
                instance_data = InstanceData()
                instance_data.bboxes = empty_bboxes
                instance_data.scores = empty_scores
                instance_data.priors = empty_bboxes.clone()
            else:
                proposals_per_img = torch.cat(proposals_list[img_id], dim=0)

                max_per_img = getattr(cfg, 'max_per_img', 1000)
                if proposals_per_img.size(0) > max_per_img:
                    scores_for_sort = proposals_per_img[:, 4]
                    _, topk_idxs = scores_for_sort.topk(max_per_img)
                    proposals_per_img = proposals_per_img[topk_idxs]

                bboxes = proposals_per_img[:, :4]
                scores = proposals_per_img[:, 4]

                instance_data = InstanceData()
                instance_data.bboxes = bboxes
                instance_data.scores = scores
                instance_data.priors = bboxes.clone()

            final_proposals.append(instance_data)

        return final_proposals

    def loss_and_proposals(self, x, batch_data_samples, proposal_cfg):
        """Forward + loss + generate proposals for training."""
        losses = self.loss(x, batch_data_samples)
        print("[loss_and_proposals] Losses:")
        for k, v in losses.items():
            if isinstance(v, list):
                print(f"  - {k}: {[i.item() for i in v]}")
            else:
                print(f"  - {k}: {v.item()}")
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
            loss_rpn_bbox_init=losses['loss_bbox_init'],
            loss_rpn_bbox_refine=losses['loss_bbox_refine']
        ), proposal_list
