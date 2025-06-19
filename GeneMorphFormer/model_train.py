import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data as data


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
        # self.pooling = nn.AdaptiveAvgPool1d(1)  # 平均池化，得到固定维度
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
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def hausdorff_loss(pred, target, alpha=2.0):
    # 计算所有点对之间的距离矩阵
    pairwise_distances = torch.cdist(pred, target, p=alpha)  # [batch_size, num_points_pred, num_points_target]

    # 每个点到目标集的最近距离
    min_distances_pred_to_target, _ = torch.min(pairwise_distances, dim=2)  # [batch_size, num_points_pred]
    min_distances_target_to_pred, _ = torch.min(pairwise_distances, dim=1)  # [batch_size, num_points_target]

    # 最大最近距离
    hausdorff_distance = torch.max(
        torch.max(min_distances_pred_to_target, dim=1)[0],  # [batch_size]
        torch.max(min_distances_target_to_pred, dim=1)[0],  # [batch_size]
    )

    return hausdorff_distance.mean()


# def compute_loss(predicted_coordinates, batch_coordinates, mask):
#     # 将第一点和最后一点的误差排除
#     mask[:, 0] = 0  # 排除第一个点
#     mask[:, -1] = 0  # 排除最后一个点
#     mse_loss = F.mse_loss(predicted_coordinates * mask.unsqueeze(-1), batch_coordinates * mask.unsqueeze(-1))
#     hausdorff = hausdorff_loss(predicted_coordinates * mask.unsqueeze(-1), batch_coordinates * mask.unsqueeze(-1))
#     loss = mse_loss * 0.8 + hausdorff * 0.6
#
#     return loss


def compute_loss(predicted_coordinates, batch_coordinates, mask, epoch, switch_epoch=200):
    # 将第一点和最后一点的误差排除
    mask[:, 0] = 0  # 排除第一个点
    mask[:, -1] = 0  # 排除最后一个点
    # 计算损失，只计算中间点
    mse_loss = F.mse_loss(predicted_coordinates * mask.unsqueeze(-1), batch_coordinates * mask.unsqueeze(-1))
    if epoch <= switch_epoch:
        loss = mse_loss
    else:
        hausdorff = hausdorff_loss(predicted_coordinates * mask.unsqueeze(-1), batch_coordinates * mask.unsqueeze(-1))
        loss = mse_loss * 0.8 + hausdorff * 0.6

    return loss


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    json_file = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/train_nor_0-600.json'
    features, positions, coordinates, mask = prepare_data_from_json(json_file)

    batch_size = 32
    dataloader = prepare_dataloader(features, positions, coordinates, mask, batch_size)

    input_dim = features.shape[2]
    model = CoordinateTransformer(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)  # 学习率减半

    save_dir = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_epoch = 1  # 默认从第1轮开始
    if os.path.exists(save_dir):
        checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('3000.ckpt')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files)
            checkpoint_path = os.path.join(save_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            for state in optimizer.state.values():
                if isinstance(state, torch.Tensor):
                    state.data = state.data.to(device)
                elif isinstance(state, dict):
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(device)

            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")

    num_epochs = 2000
    save_ckpt_interval = 400

    losses = []
    for epoch in range(start_epoch, num_epochs + start_epoch):
        model = model.to(device)
        model.train()
        epoch_loss = 0
        for batch_features, batch_positions, batch_coordinates, batch_mask in dataloader:
            batch_features, batch_positions, batch_coordinates, batch_mask = \
                batch_features.to(device), batch_positions.to(device), batch_coordinates.to(device), batch_mask.to(
                    device)
            predicted_coordinates = model(batch_features, batch_positions, batch_mask)
            # loss = compute_loss(predicted_coordinates, batch_coordinates, batch_mask)
            loss = compute_loss(predicted_coordinates, batch_coordinates, batch_mask, epoch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        epoch_loss /= len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch}/{num_epochs + start_epoch}, Loss: {epoch_loss:.6f}")

        if epoch % save_ckpt_interval == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(save_dir, f'epoch_{epoch}.ckpt'))
            print(f"Checkpoint saved at epoch {epoch}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()
