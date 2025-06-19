import os
import torch
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


# ------------------- 模型定义 -------------------
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
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

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
        self.res_blocks = nn.ModuleList([
            ResidualBlock(embed_dim, dropout) for _ in range(num_layers)
        ])

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
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(embed_dim, dropout) for _ in range(num_layers)
        ])
        self.adj = self._build_adjacency(600)

    def _build_adjacency(self, num_nodes):
        adj = torch.eye(num_nodes)
        for i in range(num_nodes):
            if i > 0: adj[i, i - 1] = 1
            if i < num_nodes - 1: adj[i, i + 1] = 1
        return adj

    def forward(self, features, positions, mask):
        x = self.input_proj(features) + self.pos_embed(positions)
        adj = self.adj.to(x.device)
        for gcn in self.gcn_layers:
            x = gcn(x, adj)
        return self._fix_coordinates(self.output_layer(x))


# ------------------- 通用函数 -------------------
# ------------------- 坐标归一化 -------------------
def normalize_coordinates(true_coords, pred_coords):
    """
    在计算损失和可视化前，对真实坐标和预测坐标进行归一化，确保它们处于相同尺度。
    """
    true_coords = np.array(true_coords)
    pred_coords = np.array(pred_coords)

    # 计算 min/max，确保所有点归一化到相同范围
    min_x, max_x = np.min(true_coords[:, 0]), np.max(true_coords[:, 0])
    min_y, max_y = np.min(true_coords[:, 1]), np.max(true_coords[:, 1])

    # 避免除零
    range_x = max_x - min_x + 1e-6
    range_y = max_y - min_y + 1e-6

    # 归一化公式
    true_coords[:, 0] = (true_coords[:, 0] - min_x) / range_x
    true_coords[:, 1] = (true_coords[:, 1] - min_y) / range_y

    pred_coords[:, 0] = (pred_coords[:, 0] - min_x) / range_x
    pred_coords[:, 1] = (pred_coords[:, 1] - min_y) / range_y

    return true_coords, pred_coords


def prepare_prediction_data(item, max_length=600):
    input_dim = 1024
    features = torch.zeros(1, max_length, input_dim)
    positions = torch.zeros(1, max_length, dtype=torch.long)
    mask = torch.zeros(1, max_length, dtype=torch.bool)

    feature_list = [point[1] for point in item]
    feature_seq = torch.tensor(feature_list)
    length = min(len(feature_seq), max_length)
    features[0, :length] = feature_seq[:length]
    positions[0, :length] = torch.arange(length)
    mask[0, :length] = True

    return features, positions, mask


# ------------------- 计算损失 -------------------
def compute_metrics(true_coords, predicted_coords):
    """计算 MSE、MAE、Hausdorff 距离和 R² 分数"""

    # 归一化坐标
    true_coords, predicted_coords = normalize_coordinates(true_coords, predicted_coords)

    # 计算误差指标
    mse = mean_squared_error(true_coords, predicted_coords)
    mae = mean_absolute_error(true_coords, predicted_coords)
    r2 = r2_score(true_coords, predicted_coords)

    # 计算 Hausdorff 距离
    true_coords_tensor = torch.tensor(true_coords, dtype=torch.float32)
    predicted_coords_tensor = torch.tensor(predicted_coords, dtype=torch.float32)

    def hausdorff_distance(pred, target):
        pairwise_distances = torch.cdist(pred, target, p=2.0)
        min_dist_pred_to_target, _ = torch.min(pairwise_distances, dim=1)
        min_dist_target_to_pred, _ = torch.min(pairwise_distances, dim=0)
        hausdorff_dist = max(torch.max(min_dist_pred_to_target), torch.max(min_dist_target_to_pred))
        return hausdorff_dist.item()

    hausdorff = hausdorff_distance(predicted_coords_tensor, true_coords_tensor)
    return mse, mae, hausdorff, r2


# ------------------- 推理主函数 -------------------
def run_inference(model_type='cnn'):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型选择
    model_classes = {
        'cnn': CoordinateCNN,
        'resnet': CoordinateResNet,
        'gcn': CoordinateGCN
    }

    # 模型路径配置
    model_paths = {
        'cnn': 'mar_gene/checkpoints_cnn/epoch_2000.ckpt',
        'resnet': 'mar_gene/checkpoints_resnet/epoch_2000.ckpt',
        'gcn': 'mar_gene/checkpoints_gcn/epoch_2000.ckpt'
    }

    # 初始化模型
    model = model_classes[model_type](input_dim=1024).to(device)

    # 加载模型参数
    checkpoint = torch.load(model_paths[model_type], map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 加载测试数据
    test_json = 'F:/Code/PythonProject/TransformerProject/New_projection/data/test_nor_0-600_test.json'
    with open(test_json, 'r') as f:
        test_data = json.load(f)

    # 输出配置
    output_folder = f'mar_gene/predicted_results_{model_type}'
    os.makedirs(output_folder, exist_ok=True)

    # 批量推理
    for idx, item in enumerate(test_data):
        segment_id = item.get("segment_id", idx)
        features, positions, mask = prepare_prediction_data(item["data"])
        features, positions, mask = features.to(device), positions.to(device), mask.to(device)

        with torch.no_grad():
            pred_coords = model(features, positions, mask).cpu().numpy().squeeze(0)

        # 获取真实坐标
        true_coordinates = np.array([point[0] for point in item["data"]])

        # 归一化真实坐标 & 预测坐标
        true_coordinates, pred_coords = normalize_coordinates(true_coordinates, pred_coords)

        # 计算误差
        mse, mae, hausdorff, r2 = compute_metrics(true_coordinates, pred_coords)

        # 保存误差信息
        metrics_output_file = os.path.join(output_folder, f"metrics_{segment_id}.json")
        with open(metrics_output_file, 'w') as f:
            json.dump({
                "segment_id": segment_id,
                "MSE": mse,
                "MAE": mae,
                "Hausdorff": hausdorff,
                "R2": r2
            }, f)

        print(f"Segment {segment_id} - MSE: {mse:.6f}, MAE: {mae:.6f}, Hausdorff: {hausdorff:.6f}, R2: {r2:.6f}")

        # **绘制归一化后的曲线**
        plt.figure(figsize=(8, 6))
        plt.plot(true_coordinates[:, 0], true_coordinates[:, 1], label='True', color='blue', marker='o')
        plt.plot(pred_coords[:, 0], pred_coords[:, 1], label='Predicted', color='red', marker='x')
        plt.xlabel("X Coordinate (Normalized)")
        plt.ylabel("Y Coordinate (Normalized)")
        plt.legend()
        plt.savefig(os.path.join(output_folder, f"predicted_plot_{segment_id}.png"))
        plt.close()


if __name__ == "__main__":
    # 选择要运行的模型类型：cnn/resnet/gcn
    run_inference(model_type='cnn')
    # run_inference(model_type='gcn')
    # run_inference(model_type='resnet')
