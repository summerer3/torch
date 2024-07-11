import os, json, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 现在可以使用PyTorch的DataLoader来加载数据集
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集。
        :param root_dir: 包含图像文件的根目录。
        :param transform: 应用于图像的可选变换。
        """
        self.root_dir = os.path.join(root_dir, '.json')
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历目录，收集图像路径和标签
        for filename in np.sort(os.listdir(root_dir)):
            if filename.endswith('.json'):  # 假设图像文件后缀为.jpg
                img_path = os.path.join('/media/liushilei/DatAset/workspace/test/torch/data/nyc/cut_data', os.path.basename(filename).split('.')[0] + '.png')
                filename = os.path.join('/media/liushilei/DatAset/workspace/test/torch/data/labels/annotation_seq', filename)
                self.images.append(img_path)
                self.labels.append(filename)

    def __len__(self):
        """
        返回数据集中的图像数量。
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        根据索引获取一个图像和它的标签。
        """
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        with open(label, 'r') as file:
            data = json.load(file)

        # 遍历JSON中的每个元素
        label = torch.zeros(100, 500, 2)
        for i, item in enumerate(data):
            seq = item.get("seq", [])
            n = len(seq)
            if n > 1:
                label[i, :n, :] = torch.tensor(seq)
        
        return image, label

H, W = 1000, 1000
class RoadEdgeDetector(nn.Module):
    def __init__(self):
        super(RoadEdgeDetector, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.rnn = nn.LSTM(input_size=4 * (H // 8) * (W // 8), hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        batch_size = x.size(0)
        features = self.backbone(x)
        features = features.view(batch_size, -1)
        features = features.unsqueeze(1)
        out, _ = self.rnn(features)
        out = self.fc(out)
        return out


def polyline_loss(pred, target):
    # 示例损失函数，可以根据实际需求调整
    loss = F.mse_loss(pred, target)
    return loss

if __name__ == "__main__":
    # 创建数据集的变换
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为Tensor
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RoadEdgeDetector().to(device)
    # 创建数据集实例
    dataset = CustomDataset(root_dir='data/labels/annotation_seq', transform=transform)
    data_loader = DataLoader(dataset, batch_size=6, shuffle=True)
    
    
    num_epochs = 50
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
 
            optimizer.zero_grad()
            outputs = model(images)
            loss = polyline_loss(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
