import torch
import torch.nn as nn
import shap
import numpy as np
import json
import os


# ----------------- 定义模型 -----------------
class CoordinateTransformer(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=1024, num_heads=8, num_layers=8, dropout=0.1, max_length=600):
        super(CoordinateTransformer, self).__init__()
        self.max_length = max_length  # 序列长度600

        self.input_embedding = nn.Linear(input_dim, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=embed_dim * 4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.coordinate_predictor = nn.Linear(embed_dim, 2)  # 输出2维坐标 (x, y)

    def forward(self, features, aggregate=False):  # aggregate 控制是否聚合
        B, L, D = features.shape
        positions = torch.arange(L).unsqueeze(0).repeat(B, 1).to(features.device)
        mask = torch.ones(B, L, dtype=torch.bool).to(features.device)

        x = self.input_embedding(features)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb

        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = x.transpose(0, 1)

        coordinates = self.coordinate_predictor(x)

        if L > 1:
            coordinates = coordinates.clone()
            coordinates[:, 0, :] = torch.tensor([0.0, 0.0], device=coordinates.device)
            coordinates[:, -1, :] = torch.tensor([600.0, 0.0], device=coordinates.device)

        # 返回 x 和 y 坐标的和，作为标量输出
        return (coordinates[:, :, 0] + coordinates[:, :, 1]).mean(dim=1) if aggregate else (
                coordinates[:, :, 0] + coordinates[:, :, 1])


# ----------------- 模型加载与初始化 -----------------
model_path = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/checkpoints/epoch_2000_2stage_mse0.8_haus0.6_1e-4_250_0.5.ckpt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 1024
model = CoordinateTransformer(input_dim)
checkpoint = torch.load(model_path, map_location=device)
model_state_dict = checkpoint.get('model_state_dict', checkpoint)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

# ----------------- 数据加载 -----------------
train_json = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/train_nor_0-600.json'
with open(train_json, 'r') as f:
    train_data = json.load(f)

train_features = np.array([np.array([point[1] for point in item["data"]]) for item in train_data])
print("Train features shape:", train_features.shape)

# 选择前50个训练样本
train_features = train_features[:50]
train_features_tensor = torch.tensor(train_features, dtype=torch.float32, requires_grad=True).to(device)

# ----------------- 数据加载 -----------------
test_json = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/test_nor_0-600_test.json'
with open(test_json, 'r') as f:
    test_data = json.load(f)

# 筛选出没有增强的数据
original_data = [entry for entry in test_data if '_reversed' not in str(entry['segment_id'])]


# 提取对应的特征
def extract_features(data):
    return np.array([np.array([point[1] for point in item["data"]]) for item in data])


all_features = extract_features(original_data)

all_features_tensor = torch.tensor(all_features, dtype=torch.float32, requires_grad=True).to(device)
print("all_features_tensor shape", all_features_tensor.shape)

# ----------------- SHAP 解释器设置 -----------------
# 用于保存 SHAP 值的目录
shap_save_dir = './shap_values/'
os.makedirs(shap_save_dir, exist_ok=True)

# ----------------- SHAP 解释器设置 -----------------
# 创建 SHAP 解释器
explainer = shap.GradientExplainer(model, train_features_tensor, batch_size=16)  # 不需要再设置 batch_size

# 用于保存 SHAP 值的目录
shap_save_dir = './shap_values/'
os.makedirs(shap_save_dir, exist_ok=True)

# 遍历 all_features_tensor 中的每个样本进行 SHAP 计算
for idx in range(all_features_tensor.shape[0]):  # 遍历每个测试样本
    # 获取单个样本数据，形状变成 [1, 600, 1024]
    sample_tensor = all_features_tensor[idx].unsqueeze(0)  # 变成 [1, 600, 1024]

    # 计算当前样本的 SHAP 值
    shap_values_batch = explainer.shap_values(sample_tensor, nsamples=64, rseed=42)
    print(f"SHAP values shape for sample {idx}: {shap_values_batch.shape}")

    # 保存每个样本的 SHAP 值
    shap_file_path = os.path.join(shap_save_dir, f"shap_values_sample_{idx}.npy")
    np.save(shap_file_path, shap_values_batch)

    # 打印保存的信息
    print(f"Saved SHAP values for sample {idx} at {shap_file_path}")
