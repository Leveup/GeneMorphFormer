import numpy as np
import json
import math
import os

def normalize_coordinates(input_file, output_file, scale_factor_val=600):
    """ 归一化坐标并存储到新 JSON 文件 """
    if not os.path.exists(input_file):
        print(f"❌ 文件未找到: {input_file}")
        return

    # 读取 JSON 数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    normalized_data = []
    for i, item in enumerate(data):
        if "data" not in item or "segment_id" not in item:
            print(f"⚠️ 跳过数据 {i}，缺少 'data' 或 'segment_id'")
            continue

        segment_id = item["segment_id"]
        segment_data = item["data"]

        if not segment_data:
            normalized_data.append({"segment_id": segment_id, "data": []})
            continue

        # 取第一个点的坐标作为参考点
        ref_x, ref_y = segment_data[0][0]

        translated_item = []
        for point in segment_data:
            original_coord = point[0]
            gene_expression = point[1]

            # 平移坐标
            new_coord = [original_coord[0] - ref_x, original_coord[1] - ref_y]
            translated_item.append([new_coord, gene_expression])

        # 获取终点坐标
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

        # 旋转和缩放
        transformed_item = []
        for point in translated_item:
            coord = point[0]
            gene_expression = point[1]

            transformed_coord = np.round(np.dot(rotation_matrix, np.array([coord[0], coord[1]])) * scale_factor, 5)
            transformed_item.append([transformed_coord.tolist(), gene_expression])

        normalized_data.append({"segment_id": segment_id, "data": transformed_item})

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存归一化数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, separators=(',', ':'), ensure_ascii=False)

    print(f"✅ 归一化完成: {output_file}")


def process_kfold_splits(data_dir, output_dir, num_folds=5):
    """ 处理 5 折交叉验证数据，执行坐标归一化 """
    os.makedirs(output_dir, exist_ok=True)

    for fold in range(1, num_folds + 1):
        train_file = os.path.join(data_dir, f"train_fold_{fold}.json")
        test_file = os.path.join(data_dir, f"test_fold_{fold}.json")

        norm_train_file = os.path.join(output_dir, f"train_fold_{fold}_normalized.json")
        norm_test_file = os.path.join(output_dir, f"test_fold_{fold}_normalized.json")

        print(f"🔹 正在处理 Fold {fold}...")

        # 归一化 train
        normalize_coordinates(train_file, norm_train_file, scale_factor_val=600)

        # 归一化 test
        normalize_coordinates(test_file, norm_test_file, scale_factor_val=600)

    print("🎉 所有折交叉验证数据归一化完成！")


if __name__ == "__main__":
    # 5 折数据目录
    data_dir = "/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/kfold_splits/"
    output_dir = "/home/work/SQT/Code/PyCharmCode/TransformerProject/New_projection/data/kfold_splits/normalized/"

    # 执行归一化
    process_kfold_splits(data_dir, output_dir)
