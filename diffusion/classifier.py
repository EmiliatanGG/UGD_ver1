import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision

# 对比学习配置
contrastive_config = {
    "projection_dim": 128,  # 对比学习投影维度
    "temperature": 0.1,  # InfoNCE温度参数
    "batch_size": 256,
    "epochs": 20,
    "lr": 3e-4
}



# 定义双分支结构
class ContrastiveModel(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.encoder = base_encoder(pretrained=False)  # 使用ResNet
        self.projector = nn.Sequential(
            nn.Linear(1000, 512),  # 假设base_encoder输出1000维
            nn.ReLU(),
            nn.Linear(512, contrastive_config["projection_dim"])
        )

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2


'''# 使用ResNet作为基础编码器
model = ContrastiveModel(torchvision.models.resnet18)
model = model.to(device)'''


# InfoNCE损失实现



# 添加分类头
class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(1000, num_classes)  # 假设encoder输出1000维

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)




