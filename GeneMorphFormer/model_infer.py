import os
import torch
import json
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ------------- 模型定义 -------------
class CoordinateTransformer(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=1024, num_heads=8, num_layers=8, dropout=0.1):
        super(CoordinateTransformer, self).__init__()

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, embed_dim)  # 将1024维特征映射到embed_dim维
        self.position_embedding = nn.Embedding(600, embed_dim)  # 假设最多600个点 # 这里有一个坐标变换

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=embed_dim * 4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全局池化和输出层
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 平均池化，得到固定维度
        self.coordinate_predictor = nn.Linear(embed_dim, 2)  # 输出2维坐标（x, y）

    def forward(self, features, positions, mask):
        """
        :param features: [B, L, input_dim] - 每个点的基因表达量
        :param positions: [B, L] - 每个点的相对位置索引
        :param mask: [B, L] - 填充部分的掩码 (1 表示有效点, 0 表示填充点)
        """
        B, L, _ = features.shape

        # 1. 输入嵌入
        x = self.input_embedding(features)  # 映射到 embed_dim, [B, L, embed_dim]

        # 添加位置编码
        pos_emb = self.position_embedding(positions)  # 位置嵌入
        x = x + pos_emb

        # 2. Transformer 编码
        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=~mask)  # 使用掩码, [L, B, embed_dim]
        x = x.transpose(0, 1)

        # 3. 输出预测
        coordinates = self.coordinate_predictor(x)  # [B, L, 2] - 每个点的2维坐标 (x, y)

        # 将第一点的坐标设定为 (0, 0)
        coordinates[:, 0, :] = torch.tensor([0.0, 0.0], device=coordinates.device)
        # 将最后一点的坐标设定为 (100, 0)
        coordinates[:, -1, :] = torch.tensor([600.0, 0.0], device=coordinates.device)

        return coordinates


# ------------- 误差计算函数 -------------
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


# ------------- 数据预处理 -------------
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

# ------------- 加载模型 -------------
model_path = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/checkpoints/epoch_2000_2stage_mse0.8_haus0.6_1e-4_250_0.5.ckpt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型并加载参数
input_dim = 1024  # 确保与训练时一致
model = CoordinateTransformer(input_dim)
# 只加载模型的参数，避免加载额外的键
checkpoint = torch.load(model_path, map_location=device)
model_state_dict = checkpoint.get('model_state_dict', checkpoint)  # 优先尝试获取'model_state_dict'，如果不存在则使用整个字典
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

# ------------- 加载测试数据 -------------
test_json = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/test_nor_0-600_test.json'
with open(test_json, 'r') as f:
    data = json.load(f)

# ------------- 输出文件夹 -------------
output_folder = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/predicted_results_2'
os.makedirs(output_folder, exist_ok=True)

# ------------- 是否计算误差的选项 -------------
compute_errors = True  # 如果不需要误差计算，可改为 False

# ------------- 批量预测 -------------
for idx, item in enumerate(data):
    segment_id = item.get("segment_id", idx)  # 获取线段 ID，若无则使用索引
    features, positions, mask = prepare_prediction_data(item["data"])
    features, positions, mask = features.to(device), positions.to(device), mask.to(device)

    with torch.no_grad():
        predicted_coordinates = model(features, positions, mask).cpu().numpy().squeeze(0)  # [L, 2]

    # 保存单个样本的预测结果（增加线段 ID 信息）
    output_file = os.path.join(output_folder, f"predicted_coordinates_{segment_id}.json")
    with open(output_file, 'w') as f:
        json.dump({
            "segment_id": segment_id,
            "predicted_coordinates": predicted_coordinates.tolist()
        }, f)

    # 获取真实坐标
    true_coordinates = np.array([point[0] for point in item["data"]])  # 提取真实坐标

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.plot(true_coordinates[:, 0], true_coordinates[:, 1], label=f'True (Segment {segment_id})',
             color='blue', marker='o', linestyle='-', markersize=5)
    plt.plot(predicted_coordinates[:, 0], predicted_coordinates[:, 1], label=f'Predicted (Segment {segment_id})',
             color='red', marker='x', linestyle='-', markersize=5)
    plt.title(f"Segment {segment_id} - True vs Predicted Coordinates")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.axis('equal')  # 设置 x 轴和 y 轴比例尺相同
    plt.savefig(os.path.join(output_folder, f"predicted_plot_{segment_id}.png"))
    plt.close()

    # 计算并输出当前样本的误差（如果启用）
    if compute_errors:
        mse, mae, hausdorff = compute_metrics(true_coordinates, predicted_coordinates)

        # 将误差信息保存到文件（包含线段 ID）
        metrics_output_file = os.path.join(output_folder, f"metrics_{segment_id}.json")
        with open(metrics_output_file, 'w') as f:
            json.dump({
                "segment_id": segment_id,
                "MSE": mse,
                "MAE": mae,
                "Hausdorff": hausdorff
            }, f)

        print(f"Segment {segment_id} - MSE: {mse:.4f}, MAE: {mae:.4f}, Hausdorff: {hausdorff:.4f}")

print(f"Predicted coordinates and plots saved to {output_folder}")

