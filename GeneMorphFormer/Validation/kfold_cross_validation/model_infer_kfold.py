import os
import torch
import json
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 模型定义保持一致
class CoordinateTransformer(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=1024, num_heads=8, num_layers=8, dropout=0.1):
        super(CoordinateTransformer, self).__init__()
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        self.position_embedding = nn.Embedding(600, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dropout=dropout, dim_feedforward=embed_dim * 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.coordinate_predictor = nn.Linear(embed_dim, 2)

    def forward(self, features, positions, mask):
        B, L, _ = features.shape
        x = self.input_embedding(features)
        x = x + self.position_embedding(positions)
        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = x.transpose(0, 1)
        coordinates = self.coordinate_predictor(x)
        coordinates[:, 0, :] = torch.tensor([0.0, 0.0], device=coordinates.device)
        coordinates[:, -1, :] = torch.tensor([600.0, 0.0], device=coordinates.device)
        return coordinates


def normalize_coordinates(true_coords, pred_coords):
    true_coords = np.array(true_coords)
    pred_coords = np.array(pred_coords)
    min_x, max_x = np.min(true_coords[:, 0]), np.max(true_coords[:, 0])
    min_y, max_y = np.min(true_coords[:, 1]), np.max(true_coords[:, 1])
    range_x = max_x - min_x + 1e-6
    range_y = max_y - min_y + 1e-6
    true_coords[:, 0] = (true_coords[:, 0] - min_x) / range_x
    true_coords[:, 1] = (true_coords[:, 1] - min_y) / range_y
    pred_coords[:, 0] = (pred_coords[:, 0] - min_x) / range_x
    pred_coords[:, 1] = (pred_coords[:, 1] - min_y) / range_y
    return true_coords, pred_coords


def compute_metrics(true_coords, predicted_coords):
    true_coords, predicted_coords = normalize_coordinates(true_coords, predicted_coords)
    mse = mean_squared_error(true_coords, predicted_coords)
    mae = mean_absolute_error(true_coords, predicted_coords)
    r2 = r2_score(true_coords, predicted_coords)
    true_tensor = torch.tensor(true_coords, dtype=torch.float32)
    pred_tensor = torch.tensor(predicted_coords, dtype=torch.float32)
    pairwise_distances = torch.cdist(pred_tensor, true_tensor, p=2.0)
    hausdorff = max(torch.max(torch.min(pairwise_distances, dim=1)[0]),
                    torch.max(torch.min(pairwise_distances, dim=0)[0])).item()
    return mse, mae, hausdorff, r2


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


# =================== 五折交叉验证评估 ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/checkpoints/'
test_data_dir = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/kfold_splits/normalized/'

mse_list, mae_list, haus_list, r2_list = [], [], [], []

for fold in range(1, 6):
    print(f"\n📦 Evaluating Fold {fold}...")
    model_path = os.path.join(model_dir, f'fold_{fold}_epoch_2000.ckpt')
    test_json = os.path.join(test_data_dir, f'test_fold_{fold}_normalized.json')

    # 初始化模型
    model = CoordinateTransformer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(test_json, 'r') as f:
        data = json.load(f)

    fold_mse, fold_mae, fold_haus, fold_r2 = [], [], [], []

    for item in data:
        features, positions, mask = prepare_prediction_data(item["data"])
        features, positions, mask = features.to(device), positions.to(device), mask.to(device)
        with torch.no_grad():
            pred_coords = model(features, positions, mask).cpu().numpy().squeeze(0)
        true_coords = np.array([point[0] for point in item["data"]])
        mse, mae, haus, r2 = compute_metrics(true_coords, pred_coords)
        fold_mse.append(mse)
        fold_mae.append(mae)
        fold_haus.append(haus)
        fold_r2.append(r2)

    mean_mse = np.mean(fold_mse)
    mean_mae = np.mean(fold_mae)
    mean_haus = np.mean(fold_haus)
    mean_r2 = np.mean(fold_r2)

    print(f"Fold {fold} - MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, Hausdorff: {mean_haus:.4f}, R²: {mean_r2:.4f}")

    mse_list.append(mean_mse)
    mae_list.append(mean_mae)
    haus_list.append(mean_haus)
    r2_list.append(mean_r2)

# =================== 汇总统计结果 ===================
print("\n🎯 Five-Fold Cross-Validation Summary:")
print(f"MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
print(f"MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"Hausdorff: {np.mean(haus_list):.4f} ± {np.std(haus_list):.4f}")
print(f"R²: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
