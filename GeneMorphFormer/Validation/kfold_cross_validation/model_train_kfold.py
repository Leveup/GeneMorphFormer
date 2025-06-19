import torch.nn as nn
import torch.nn.functional as F
import json
import os
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt


class CoordinateTransformer(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=1024, num_heads=8, num_layers=8, dropout=0.1):
        super(CoordinateTransformer, self).__init__()

        self.input_embedding = nn.Linear(input_dim, embed_dim)
        self.position_embedding = nn.Embedding(600, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=embed_dim * 4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.coordinate_predictor = nn.Linear(embed_dim, 2)

    def forward(self, features, positions, mask):
        B, L, _ = features.shape

        x = self.input_embedding(features)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb

        x = x.transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = x.transpose(0, 1)

        coordinates = self.coordinate_predictor(x)

        coordinates[:, 0, :] = torch.tensor([0.0, 0.0], device=coordinates.device)
        coordinates[:, -1, :] = torch.tensor([600.0, 0.0], device=coordinates.device)

        return coordinates


def prepare_data_from_json(json_file, max_length=600):
    with open(json_file, 'r') as f:
        data = json.load(f)

    batch_size = len(data)
    input_dim = 1024

    features = torch.zeros(batch_size, max_length, input_dim)
    coordinates = torch.zeros(batch_size, max_length, 2)
    positions = torch.zeros(batch_size, max_length, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for i, item in enumerate(data):
        feature_list = []
        coord_list = []

        for point in item["data"]:
            coord = point[0]
            gene_expression = point[1]

            coord_list.append(coord)
            feature_list.append(gene_expression)

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
    pairwise_distances = torch.cdist(pred, target, p=alpha)

    min_distances_pred_to_target, _ = torch.min(pairwise_distances, dim=2)
    min_distances_target_to_pred, _ = torch.min(pairwise_distances, dim=1)

    hausdorff_distance = torch.max(
        torch.max(min_distances_pred_to_target, dim=1)[0],
        torch.max(min_distances_target_to_pred, dim=1)[0],
    )

    return hausdorff_distance.mean()


def compute_loss(predicted_coordinates, batch_coordinates, mask, epoch, switch_epoch=200):
    mask[:, 0] = 0
    mask[:, -1] = 0
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

    data_dir = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/kfold_splits/normalized/'
    save_dir = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 32
    num_epochs = 1000
    save_ckpt_interval = 400
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files = sorted([f for f in os.listdir(data_dir) if f.startswith("train_fold") and f.endswith("_normalized.json")])
    test_files = sorted([f.replace("train", "test") for f in train_files])

    fold_losses = []

    for fold, (train_file, test_file) in enumerate(zip(train_files, test_files), 1):
        print(f"Starting Fold {fold} Training...")

        train_features, train_positions, train_coordinates, train_mask = prepare_data_from_json(os.path.join(data_dir, train_file))
        test_features, test_positions, test_coordinates, test_mask = prepare_data_from_json(os.path.join(data_dir, test_file))

        train_loader = prepare_dataloader(train_features, train_positions, train_coordinates, train_mask, batch_size)
        test_loader = prepare_dataloader(test_features, test_positions, test_coordinates, test_mask, batch_size)

        model = CoordinateTransformer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)

        fold_train_losses = []
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_loss = 0
            for batch_features, batch_positions, batch_coordinates, batch_mask in train_loader:
                batch_features, batch_positions, batch_coordinates, batch_mask = \
                    batch_features.to(device), batch_positions.to(device), batch_coordinates.to(device), batch_mask.to(device)

                predicted_coordinates = model(batch_features, batch_positions, batch_mask)
                loss = compute_loss(predicted_coordinates, batch_coordinates, batch_mask, epoch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            epoch_loss /= len(train_loader)
            fold_train_losses.append(epoch_loss)
            print(f"Fold {fold}, Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.6f}")

            if epoch % save_ckpt_interval == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(save_dir, f'fold_{fold}_epoch_{epoch}.ckpt'))
                print(f"Checkpoint saved at epoch {epoch}")

        fold_losses.append(fold_train_losses)

    # 画出五折交叉验证的损失曲线
    plt.figure(figsize=(10, 5))
    for fold_idx, losses in enumerate(fold_losses, 1):
        plt.plot(losses, label=f"Fold {fold_idx}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve for 5-Fold Cross Validation")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "cross_validation_loss.png"))
    plt.show()
