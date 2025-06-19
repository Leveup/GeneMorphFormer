import scipy.io as sio
import json
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler

def read_line_segments(txt_file):
    line_segments = []
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(":")
                    if len(parts) != 2:
                        raise ValueError(f"数据格式错误: {line}")
                    segment = parts[1].strip()
                    segment_numbers = segment.split()
                    if not all(num.isdigit() for num in segment_numbers):
                        raise ValueError(f"非数字字符: {segment}")
                    segment_numbers = list(map(int, segment_numbers))
                    line_segments.append(segment_numbers)
    except FileNotFoundError:
        print(f"文件 {txt_file} 不存在，请检查文件路径。")
    return line_segments


def load_mat_file(mat_file1, mat_file2):
    mat_data1 = sio.loadmat(mat_file1)
    with h5py.File(mat_file2, 'r') as f:
        expression_matrix = f['adjusted_expression_matrix'][:].T
        print("Original shape of expression_matrix:", expression_matrix.shape)

    this_coord = mat_data1['coord']
    scaler = StandardScaler()
    expression_matrix_nor = np.array([
        np.round(scaler.fit_transform(row.reshape(-1, 1)).flatten(), 5)
        for row in expression_matrix
    ])
    return this_coord, expression_matrix_nor


def extract_coord_and_expression(line_segments, this_coord, expression_matrix):
    data = []
    expression_matrix = expression_matrix.T  # 匹配点顺序
    for idx, segment in enumerate(line_segments):
        segment = [s - 1 for s in segment]  # 索引从0开始
        coord_data = this_coord[segment]
        expression_data = expression_matrix[segment]

        combined_data = {
            "segment_id": idx + 1,
            "data": list(zip(coord_data.tolist(), expression_data.tolist()))
        }
        data.append(combined_data)
    return data


def data_augmentation(data):
    augmented_data = []
    for segment_data in data:
        augmented_data.append(segment_data)
        reversed_data = {
            "segment_id": f"{segment_data['segment_id']}_reversed",
            "data": segment_data["data"][::-1]
        }
        augmented_data.append(reversed_data)
    return augmented_data


# 动态识别线段连续组，基于每条线段的起始点索引是否连续判断分组
def detect_groups(line_segments):
    """
    根据线段之间是否“终点+1=下个线段起点”来划分连续组
    """
    segment_limits = [(segment[0], segment[-1]) for segment in line_segments]

    groups = []
    current_group = [0]  # 存储当前组线段的索引，初始第一个线段索引为0
    for i in range(1, len(segment_limits)):
        prev_end = segment_limits[i - 1][1]
        curr_start = segment_limits[i][0]

        if curr_start == prev_end + 1:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    if current_group:
        groups.append(current_group)

    print(f"共检测到 {len(groups)} 个连续线段组（半脑）")
    for i, group in enumerate(groups):
        start_point = segment_limits[group[0]][0]
        end_point = segment_limits[group[-1]][1]
        print(f"组 {i + 1} 包含 {len(group)} 条线段，起始点：{start_point}，结束点：{end_point}")

    return groups


def split_by_detected_groups(data, groups, test_group_count=2):
    train_indices = []
    test_indices = []
    for i, group in enumerate(groups):
        if i >= len(groups) - test_group_count:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    return train_data, test_data


def save_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, separators=(',', ':'), ensure_ascii=False)


def main(txt_file, mat_file1, mat_file2, output_json_1, output_json_2):
    # 读取线段编号
    line_segments = read_line_segments(txt_file)

    # 读取坐标和表达数据（已标准化）
    this_coord, expression_matrix = load_mat_file(mat_file1, mat_file2)
    print('坐标点数:', len(this_coord), '基因表达数据点数:', len(expression_matrix))

    # 提取每条线段对应的坐标 + 表达量数据
    data = extract_coord_and_expression(line_segments, this_coord, expression_matrix)

    # 检测连续的“半脑组”（线段连续性）
    groups = detect_groups(line_segments)

    # 划分训练组和测试组（默认保留最后2个组为测试集）
    train_data, test_data = split_by_detected_groups(data, groups, test_group_count=2)

    # 各自进行数据增强
    train_data = data_augmentation(train_data)
    test_data = data_augmentation(test_data)

    # 保存为 JSON
    save_to_json(train_data, output_json_1)
    print(f"训练数据已保存到 {output_json_1}")
    save_to_json(test_data, output_json_2)
    print(f"测试数据已保存到 {output_json_2}")


txt_file = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/segments.txt'
mat_file1 = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/all.mat'
mat_file2 = '/home/data/SQT/Extract_gyri_sulci_curv/Marmoset_extract_gene_feature_by_nissl_line' \
            '/Marmoset_Processed_Genes_feature_Smooth/adjusted_expression_matrix.mat'
output_json_1 = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/check_for_val/data/train_data.json'
output_json_2 = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/check_for_val/data/test_data.json'

main(txt_file, mat_file1, mat_file2, output_json_1, output_json_2)
