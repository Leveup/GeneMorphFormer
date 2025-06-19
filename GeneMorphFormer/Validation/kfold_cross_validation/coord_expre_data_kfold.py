import os

import scipy.io as sio
import json
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# 读取线段索引
def read_line_segments(txt_file):
    line_segments = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(":")
                if len(parts) != 2:
                    raise ValueError(f"数据格式错误: {line}")
                segment = parts[1].strip().split()
                segment_numbers = list(map(int, segment))
                line_segments.append(segment_numbers)
    return line_segments


# 读取 .mat 文件数据
def load_mat_file(mat_file1, mat_file2):
    mat_data1 = sio.loadmat(mat_file1)
    with h5py.File(mat_file2, 'r') as f:
        expression_matrix = f['adjusted_expression_matrix'][:].T

    this_coord = mat_data1['coord']
    scaler = StandardScaler()
    expression_matrix_nor = np.array([
        np.round(scaler.fit_transform(row.reshape(-1, 1)).flatten(), 5)
        for row in expression_matrix
    ])

    return this_coord, expression_matrix_nor


# 提取坐标和基因表达数据
def extract_coord_and_expression(line_segments, this_coord, expression_matrix):
    data = []
    expression_matrix = expression_matrix.T  # 转置以匹配点顺序
    for idx, segment in enumerate(line_segments):
        segment = [s - 1 for s in segment]  # 0-based 索引
        coord_data = this_coord[segment]
        expression_data = expression_matrix[segment]

        combined_data = {
            "segment_id": idx + 1,
            "data": list(zip(coord_data.tolist(), expression_data.tolist()))
        }
        data.append(combined_data)
    return data


# 数据增强（添加逆序版本）
def data_augmentation(data):
    augmented_data = []
    for segment_data in data:
        augmented_data.append(segment_data)
        reversed_data = {
            "segment_id": f"{segment_data['segment_id']}_reversed",
            "data": segment_data["data"][::-1]  # 逆序
        }
        augmented_data.append(reversed_data)
    return augmented_data


# 5 折交叉验证数据划分
def kfold_split(data, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = list(kf.split(data))  # 生成 K 折索引
    return folds


# 保存 JSON 数据
def save_to_json(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 确保目录存在
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ':'), ensure_ascii=False)


# 主函数：进行 5 折交叉验证
def main(txt_file, mat_file1, mat_file2, output_dir, k=5):
    line_segments = read_line_segments(txt_file)
    this_coord, expression_matrix = load_mat_file(mat_file1, mat_file2)

    # 生成数据
    data = extract_coord_and_expression(line_segments, this_coord, expression_matrix)
    augmented_data = data_augmentation(data)  # 增强数据

    # 5 折交叉验证划分
    folds = kfold_split(augmented_data, k)

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_data = [augmented_data[i] for i in train_idx]
        test_data = [augmented_data[i] for i in test_idx]

        train_json = f"{output_dir}/train_fold_{fold_idx+1}.json"
        test_json = f"{output_dir}/test_fold_{fold_idx+1}.json"

        save_to_json(train_data, train_json)
        save_to_json(test_data, test_json)

        print(f"Fold {fold_idx+1}: 训练数据 {len(train_data)} 条, 测试数据 {len(test_data)} 条")
        print(f"训练集保存至: {train_json}")
        print(f"测试集保存至: {test_json}")


# 文件路径
txt_file = "/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/segments.txt"
mat_file1 = "/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/all.mat"
mat_file2 = "/home/data/SQT/Extract_gyri_sulci_curv/Marmoset_extract_gene_feature_by_nissl_line" \
            "/Marmoset_Processed_Genes_feature_Smooth/adjusted_expression_matrix.mat"
output_dir = "/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/kfold_splits"

# 执行 5 折交叉验证数据划分
main(txt_file, mat_file1, mat_file2, output_dir, k=5)
