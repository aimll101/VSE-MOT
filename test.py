import sys
sys.path.append('/home/jundu/MOTRv2-r1/models')
from image_encoder import ImageEncoder
import torch
import torch
import torch.nn as nn
import clip


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # 加载 CLIP 模型，选择 ResNet50 结构的图像编码器
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, self.preprocess = clip.load("RN50", device=device)
        # 只保留图像编码器
        self.image_encoder = clip_model.visual
        self.image_encoder.eval()

    def forward(self, image_features):
        # 假设输入的 image_features 是形状为 (C, H, W) 的张量
        # 首先将其添加一个批次维度，使其形状变为 (1, C, H, W)
        image_features = image_features.unsqueeze(0)
        # 确保图像特征在正确的设备上
        device = next(self.image_encoder.parameters()).device
        image_features = image_features.to(device)
        # 使用 CLIP 的图像编码器提取特征
        with torch.no_grad():
            image_features = self.image_encoder(image_features)
        # 调整特征形状为 (1, 1024)
        return image_features
custom_model = ImageEncoder()
total_params = sum(p.numel() for p in custom_model.parameters() if p.requires_grad)
print(f"模型的参数量为: {total_params}")
# 假设我们有一个形状为 (C, H, W) 的图像特征
# 这里我们随机生成一个示例
C, H, W = 3, 224, 224
image_features = torch.randn(C, H, W)
# 处理图像特征
processed_features = custom_model(image_features)
print("处理后的特征形状:", processed_features.shape)