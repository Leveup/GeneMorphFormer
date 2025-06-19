import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import os
import torch.utils.data as data
import numpy as np


# -------------------- 数据加载模块（保持不变） --------------------
def prepare_data_from_json(json_file, max_length=600):  # 这里有一个坐标变换 600 个点
    with open(json_file, 'r') as f:
        data = json.load(f)

    batch_size = len(data)
    input_dim = 1024  # 基因表达量的维度

    features = torch.zeros(batch_size, max_length, input_dim)
    coordinates = torch.zeros(batch_size, max_length, 2)
    positions = torch.zeros(batch_size, max_length, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    # 初始化标准化器
    # scaler = StandardScaler()

    for i, item in enumerate(data):
        # 每个点的数据格式：[坐标, 基因表达量]
        feature_list = []
        coord_list = []

        for point in item["data"]:  # 结构是segment_id + data
            coord = point[0]  # 假设坐标是列表的前两个元素
            gene_expression = point[1]  # 基因表达量

            # 分离坐标和基因表达量
            coord_list.append(coord)   # [x, y]
            feature_list.append(gene_expression)  # 基因表达量

        feature_seq = torch.tensor(feature_list)
        coord_seq = torch.tensor(coord_list)

        length = min(len(feature_seq), max_length)
        features[i, :length] = feature_seq[:length]
        coordinates[i, :length] = coord_seq[:length]
        positions[i, :length] = torch.arange(length)
        mask[i, :length] = True

    return features, positions, coordinates, mask


def prepare_dataloader(features, positions, coordinates, mask, batch_size):
    dataset = data.TensorDataset(features, positions, coordinates, mask)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# -------------------- 模型定义模块 --------------------
class BaseModel(nn.Module):
    """模型基类，包含通用组件"""

    def __init__(self, input_dim=1024, embed_dim=1024):
        super().__init__()
        self.pos_embed = nn.Embedding(600, embed_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.output_layer = nn.Linear(embed_dim, 2)

    def _fix_coordinates(self, coords):
        """固定首尾坐标"""
        coords[:, 0] = torch.tensor([0.0, 0.0], device=coords.device)
        coords[:, -1] = torch.tensor([600.0, 0.0], device=coords.device)
        return coords


class CoordinateCNN(BaseModel):
    """CNN架构"""

    def __init__(self, input_dim=1024, embed_dim=1024, num_layers=8, dropout=0.1):
        super().__init__(input_dim, embed_dim)

        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))

    def forward(self, features, positions, mask):
        x = self.input_proj(features) + self.pos_embed(positions)
        x = x.transpose(1, 2)  # [B, D, L]

        for conv in self.conv_layers:
            x = conv(x)

        x = x.transpose(1, 2)  # [B, L, D]
        return self._fix_coordinates(self.output_layer(x))

class ResidualBlock(nn.Module):
    """ResNet残差块"""

    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)

class CoordinateResNet(BaseModel):
    """ResNet架构"""

    def __init__(self, input_dim=1024, embed_dim=1024, num_layers=8, dropout=0.1):
        super().__init__(input_dim, embed_dim)

        self.res_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.res_blocks.append(ResidualBlock(embed_dim, dropout))

    def forward(self, features, positions, mask):
        x = self.input_proj(features) + self.pos_embed(positions)
        x = x.transpose(1, 2)  # [B, D, L]

        for block in self.res_blocks:
            x = block(x)

        x = x.transpose(1, 2)  # [B, L, D]
        return self._fix_coordinates(self.output_layer(x))

class GraphConvLayer(nn.Module):
    """图卷积层"""

    def __init__(self, dim, dropout):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        agg = torch.matmul(adj.float(), x)  # 邻居聚合
        agg = self.linear(agg)
        x = self.norm(x + agg)
        return self.dropout(F.gelu(x))

class CoordinateGCN(BaseModel):
    """GCN架构"""

    def __init__(self, input_dim=1024, embed_dim=1024, num_layers=8, dropout=0.1):
        super().__init__(input_dim, embed_dim)

        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn_layers.append(GraphConvLayer(embed_dim, dropout))

        # 构建线性邻接矩阵
        self.adj = self._build_adjacency(600)

    def _build_adjacency(self, num_nodes):
        adj = torch.eye(num_nodes)
        for i in range(num_nodes):
            if i > 0: adj[i, i - 1] = 1
            if i < num_nodes - 1: adj[i, i + 1] = 1
        return adj  # [L, L]

    def forward(self, features, positions, mask):
        x = self.input_proj(features) + self.pos_embed(positions)
        adj = self.adj.to(x.device)

        for gcn in self.gcn_layers:
            x = gcn(x, adj)

        return self._fix_coordinates(self.output_layer(x))

# -------------------- 训练模块 --------------------
def hausdorff_loss(pred, target, alpha=2.0):
    pairwise_dist = torch.cdist(pred, target, p=alpha)
    min_pred = torch.min(pairwise_dist, dim=2)[0]
    min_target = torch.min(pairwise_dist, dim=1)[0]
    return torch.max(torch.max(min_pred, dim=1)[0], torch.max(min_target, dim=1)[0]).mean()


def compute_loss(pred, target, mask, epoch):
    mask[:, [0, -1]] = 0  # 排除首尾点
    mse = F.mse_loss(pred * mask.unsqueeze(-1), target * mask.unsqueeze(-1))
    if epoch > 200:  # 200 epoch后加入hausdorff损失
        return mse * 0.8 + hausdorff_loss(pred * mask.unsqueeze(-1), target * mask.unsqueeze(-1)) * 0.6
    return mse

def train_model(model_type='cnn'):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据准备
    features, positions, coordinates, mask = prepare_data_from_json('/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/train_nor_0-600.json')
    dataloader = prepare_dataloader(features, positions, coordinates, mask, batch_size=32)

    # 模型初始化
    model_classes = {
        'cnn': CoordinateCNN,
        'resnet': CoordinateResNet,
        'gcn': CoordinateGCN
    }
    model = model_classes[model_type](input_dim=1024).to(device)

    # 优化配置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 250, 0.5)

    # 训练循环
    losses = []
    num_epochs = 2000
    save_ckpt_interval = 400
    save_dir = f'mar_gene/checkpoints_{model_type}'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0

        for batch in dataloader:
            features, pos, coords, mask = [x.to(device) for x in batch]

            optimizer.zero_grad()
            pred = model(features, pos, mask)
            loss = compute_loss(pred, coords, mask, epoch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f}')

        # 保存检查点
        if epoch % save_ckpt_interval == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }, f'{save_dir}/epoch_{epoch}.ckpt')

    # 绘制损失曲线
    # plt.figure(figsize=(10, 6))
    # plt.plot(losses, label=f'{model_type.upper()} Training Loss')
    # plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('Training Curve')
    # plt.legend(), plt.grid(True)
    # plt.savefig(f'{model_type}_loss_curve.png')
    # plt.show()

if __name__ == "__main__":
    # 选择要训练的模型类型：cnn/resnet/gcn
    # train_model(model_type='gcn')  # 修改此处切换模型
    train_model(model_type='cnn')
    train_model(model_type='resnet')