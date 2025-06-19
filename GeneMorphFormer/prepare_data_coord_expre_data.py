import scipy.io as sio
import json
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
import random

# 读取线段索引的 txt 文件
def read_line_segments(txt_file):
    line_segments = []
    try:
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()  # 去除空白字符
                if line:
                    parts = line.split(":")
                    if len(parts) != 2:
                        raise ValueError(f"数据格式错误: {line}")
                    segment = parts[1].strip()  # 提取线段数据部分
                    segment_numbers = segment.split()
                    if not all(num.isdigit() for num in segment_numbers):
                        raise ValueError(f"非数字字符: {segment}")
                    segment_numbers = list(map(int, segment_numbers))
                    line_segments.append(segment_numbers)
    except FileNotFoundError:
        print(f"文件 {txt_file} 不存在，请检查文件路径。")
    return line_segments


# 加载 .mat 文件中的坐标和基因表达数据
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


# 提取坐标和基因表达数据，并附加 segment_id
def extract_coord_and_expression(line_segments, this_coord, expression_matrix):
    data = []
    expression_matrix = expression_matrix.T  # 转置以匹配点顺序
    for idx, segment in enumerate(line_segments):
        segment = [s - 1 for s in segment]  # 调整索引从0开始
        coord_data = this_coord[segment]
        expression_data = expression_matrix[segment]

        # 转换为列表
        coord_data = coord_data.tolist()
        expression_data = expression_data.tolist()

        # 组合数据并附加 segment_id
        combined_data = {
            "segment_id": idx + 1,  # 线段编号从1开始
            "data": list(zip(coord_data, expression_data))
        }

        data.append(combined_data)
    return data


# 数据增强（添加逆序版本）
def data_augmentation(data):
    augmented_data = []
    for segment_data in data:
        augmented_data.append(segment_data)

        # 逆序线段数据，并保留 segment_id，给逆序数据分配新的 segment_id
        reversed_data = {
            "segment_id": f"{segment_data['segment_id']}_reversed",  # 在 segment_id 后添加标记
            "data": segment_data["data"][::-1]  # 逆序操作
        }
        augmented_data.append(reversed_data)

    return augmented_data


# 划分数据集
def split_data(data, test_ratio=0.1):
    # random.shuffle(data)
    test_size = int(len(data) * test_ratio)
    test_set = data[:test_size]
    train_set = data[test_size:]
    return train_set, test_set


# 保存 JSON 数据
def save_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ':'), ensure_ascii=False)


# 主函数
def main(txt_file, mat_file1, mat_file2, output_json_1, output_json_2):
    # 读取线段数据
    line_segments = read_line_segments(txt_file)

    # 加载坐标和基因表达数据
    this_coord, expression_matrix = load_mat_file(mat_file1, mat_file2)
    print('坐标点数:', len(this_coord), '基因表达数据点数:', len(expression_matrix))

    # 提取数据并附加 segment_id
    data = extract_coord_and_expression(line_segments, this_coord, expression_matrix)

    # 数据增强
    augmented_data = data_augmentation(data)

    # 划分数据集
    train_data, test_data = split_data(augmented_data)

    # 保存数据到 JSON
    save_to_json(train_data, output_json_1)
    print(f"训练数据已保存到 {output_json_1}")
    save_to_json(test_data, output_json_2)
    print(f"测试数据已保存到 {output_json_2}")


# 设置文件路径
txt_file = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/segments.txt'

mat_file1 = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/all.mat'
mat_file2 = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/adjusted_expression_matrix.mat'

output_json_1 = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/train_nor_norandom.json'
output_json_2 = '/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/test_nor_norandom.json'

# 执行主函数
main(txt_file, mat_file1, mat_file2, output_json_1, output_json_2)
