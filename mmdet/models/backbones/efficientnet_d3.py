from mmdet.registry import BACKBONES
from mmcv.runner import BaseModule

@BACKBONES.register_module()
class EfficientNetD3Backbone(BaseModule):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load EfficientNet-D3
        from efficientnet_pytorch import EfficientNet
        self.model = EfficientNet.from_pretrained("efficientnet-b3") if pretrained else EfficientNet.from_name("efficientnet-b3")
    
    def forward(self, x):
        features = self.model.extract_endpoints(x)
        return [features['reduction_3'], features['reduction_4'], features['reduction_5']]
