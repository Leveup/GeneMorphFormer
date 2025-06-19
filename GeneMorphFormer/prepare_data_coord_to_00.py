import numpy as np
import json
import math


def normalize_coordinates(input_file, output_file, scale_factor_val, selected_indices=None):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    normalized_data = []
    for i, item in enumerate(data):
        if "data" not in item or "segment_id" not in item:
            print(f"Skipping item {i}, missing 'data' or 'segment_id' key")
            continue

        segment_id = item["segment_id"]  # 提取线段 ID
        segment_data = item["data"]

        if not segment_data:
            normalized_data.append({"segment_id": segment_id, "data": []})  # 处理空线段
            continue

        # 取第一个点的坐标作为参考点
        ref_x, ref_y = segment_data[0][0]

        translated_item = []
        for point in segment_data:
            original_coord = point[0]  # 获取坐标
            gene_expression = point[1]  # 获取基因表达量

            # 平移坐标
            new_coord = [original_coord[0] - ref_x, original_coord[1] - ref_y]
            translated_item.append([new_coord, gene_expression])

        # 获取平移后的终点坐标
        x1, y1 = translated_item[0][0]
        x2, y2 = translated_item[-1][0]

        # 计算线段长度并缩放
        original_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        scale_factor = scale_factor_val / original_length

        # 计算旋转角度
        angle = math.atan2(y2 - y1, x2 - x1)

        # 计算旋转矩阵
        rotation_matrix = np.array([[math.cos(-angle), -math.sin(-angle)],
                                    [math.sin(-angle), math.cos(-angle)]])

        # 旋转和缩放坐标
        transformed_item = []
        for point in translated_item:
            coord = point[0]
            gene_expression = point[1]

            # 旋转并缩放坐标
            transformed_coord = np.round(np.dot(rotation_matrix, np.array([coord[0], coord[1]])) * scale_factor, 5)
            transformed_item.append([transformed_coord.tolist(), gene_expression])

        # 选择特定的索引或存入所有数据，并保留segment_id
        if selected_indices is None or i in selected_indices:
            normalized_data.append({"segment_id": segment_id, "data": transformed_item})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, separators=(',', ':'), ensure_ascii=False)

    print("Normalization complete and saved to", output_file)


# 选择特定的线段索引
scale_factor_val = 600  # 缩放大小
selected_indices = None  # 指定要筛选的线段，如果要处理全部数据，设为空列表 []
# input_file = 'F:/Code/PythonProject/TransformerProject/New_projection/data/train_nor.json'
# output_file = 'F:/Code/PythonProject/TransformerProject/New_projection/data/train_nor_0-600.json'

input_file = 'F:/Code/PythonProject/TransformerProject/New_projection/data/test_nor.json'
output_file = 'F:/Code/PythonProject/TransformerProject/New_projection/data/test_nor_0-600_test.json'

normalize_coordinates(input_file, output_file, scale_factor_val, selected_indices)
