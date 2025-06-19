import os
import torch
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
# def hausdorff_loss(pred, target, alpha=2.0):
#     distances_pred_to_target = torch.cdist(pred, target, p=alpha)
#     min_dist_pred = torch.min(distances_pred_to_target, dim=1)[0]
#     min_dist_target = torch.min(distances_pred_to_target, dim=2)[0]
#     return torch.max(torch.max(min_dist_pred), torch.max(min_dist_target))

def hausdorff_loss(pred, target, alpha=2.0):
    # 计算 pred 中每个点到 target 中每个点的距离
    distances_pred_to_target = torch.cdist(pred, target, p=alpha)
    # 找到 pred 中每个点到 target 的最近距离
    min_distances_pred_to_target = torch.min(distances_pred_to_target, dim=1)[0]
    # 计算 target 中每个点到 pred 中每个点的距离
    distances_target_to_pred = torch.cdist(target, pred, p=alpha)
    # 找到 target 中每个点到 pred 的最近距离
    min_distances_target_to_pred = torch.min(distances_target_to_pred, dim=1)[0]
    # 最大的最近距离作为豪斯多夫距离
    hausdorff_distance = torch.max(
        torch.max(min_distances_pred_to_target), torch.max(min_distances_target_to_pred)
    )
    return hausdorff_distance

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


def compute_metrics(true_coords, predicted_coords):
    mse = mean_squared_error(true_coords, predicted_coords)
    mae = mean_absolute_error(true_coords, predicted_coords)

    # 计算 Hausdorff 距离
    true_coords_tensor = torch.tensor(true_coords)
    predicted_coords_tensor = torch.tensor(predicted_coords)

    # 确保两个张量的数据类型一致
    true_coords_tensor = true_coords_tensor.to(predicted_coords_tensor.dtype)

    def hausdorff_distance(pred, target):
        pairwise_distances = torch.cdist(pred, target, p=2.0)  # 欧几里得距离
        min_dist_pred_to_target, _ = torch.min(pairwise_distances, dim=1)
        min_dist_target_to_pred, _ = torch.min(pairwise_distances, dim=0)
        hausdorff_dist = max(torch.max(min_dist_pred_to_target), torch.max(min_dist_target_to_pred))
        return hausdorff_dist.item()

    hausdorff = hausdorff_distance(predicted_coords_tensor, true_coords_tensor)

    return mse, mae, hausdorff

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
    test_json = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/test_nor_0-600_test.json'
    with open(test_json, 'r') as f:
        test_data = json.load(f)

    # 输出配置
    output_folder = f'mar_gene/predicted_results_{model_type}'
    os.makedirs(output_folder, exist_ok=True)

    compute_errors = True  # 如果不需要误差计算，可改为 False
    # 批量推理
    for idx, item in enumerate(test_data):
        segment_id = item.get("segment_id", idx)  # 获取线段 ID，若无则使用索引
        features, positions, mask = prepare_prediction_data(item["data"])
        features, positions, mask = features.to(device), positions.to(device), mask.to(device)

        with torch.no_grad():
            pred_coords = model(features, positions, mask).cpu().numpy().squeeze(0)
        true_coordinates = np.array([point[0] for point in item["data"]])  # 提取真实坐标

        # 计算并输出当前样本的误差（如果启用）
        if compute_errors:
            mse, mae, hausdorff = compute_metrics(true_coordinates, pred_coords)

            # 将误差信息保存到文件（包含线段 ID）
            metrics_output_file = os.path.join(output_folder, f"metrics_{segment_id}.json")
            with open(metrics_output_file, 'w') as f:
                json.dump({
                    "Segment_id": segment_id,
                    "MSE": mse,
                    "MAE": mae,
                    "Hausdorff": hausdorff
                }, f)

            print(f"Segment {segment_id} - MSE: {mse:.4f}, MAE: {mae:.4f}, Hausdorff: {hausdorff:.4f}")

        plt.figure(figsize=(8, 6))
        plt.plot(true_coordinates[:, 0], true_coordinates[:, 1], label='True Coordinates', color='blue', marker='o',
                 linestyle='-', markersize=5)
        plt.plot(pred_coords[:, 0], pred_coords[:, 1], label='Predicted Coordinates', color='red',
                 marker='x', linestyle='-', markersize=5)
        plt.title(f"Sample {segment_id} - True vs Predicted Coordinates")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.axis('equal')  # 设置 x 轴和 y 轴比例尺相同
        plt.savefig(os.path.join(output_folder, f"predicted_plot_{segment_id}.png"))
        plt.close()

if __name__ == "__main__":
    # 选择要运行的模型类型：cnn/resnet/gcn
    # run_inference(model_type='cnn')
    # run_inference(model_type='gcn')
    run_inference(model_type='resnet')
