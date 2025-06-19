import scipy.io

# 加载.mat文件
mat_data = scipy.io.loadmat('F:/Code/PythonProject/TransformerProject/New_projection/data/all.mat')
# 替换为你的.mat文件名

if 'coord' in mat_data:
    all_coords = mat_data['coord']
    # 检查矩阵的大小
    print(f"原始all_coords矩阵大小: {all_coords.shape}")

# 提取名为'all_coords'的矩阵
if 'coord' in mat_data:
    points = mat_data['coord'].astype(float)  # 获取'all_coords'矩阵
else:
    raise ValueError("文件中未找到名为'coord'的矩阵！")

if points.shape[1]!= 2:
    raise ValueError(f"'coord'的维度为 {points.shape}，但应为 (51916, 2)")

segments = []  # 用于存储所有线段
current_segment = [1]  # 当前线段（初始从第一个点开始）

for i in range(1, len(points)):
    # 获取相邻点之间的x和y差值
    dx = abs(points[i, 0] - points[i - 1, 0])
    dy = abs(points[i, 1] - points[i - 1, 1])

    # 判断是否属于同一线段
    if dx < 5 and dy < 5:
        current_segment.append(i + 1)  # 点序号从1开始
    else:
        # 保存当前线段并开始新的线段
        segments.append(current_segment)
        current_segment = [i + 1]

if current_segment:
    segments.append(current_segment)

# 拆分超过600点的线段，并过滤掉点数小于 num_point 的线段
num_point = 600
processed_segments = []
for segment in segments:
    if len(segment) >= num_point:
        # 将长线段拆分为多个小段，每段最多100个点
        for i in range(0, len(segment), num_point):
            sub_segment = segment[i:i + num_point]
            if (len(segment) - i) >= num_point:
                processed_segments.append(sub_segment)

# 保存结果到文本文件
output_file = 'F:/Code/PythonProject/TransformerProject/New_projection/data/segments.txt'
with open(output_file, 'w') as f:
    for idx, segment in enumerate(processed_segments):
        segment_str = ' '.join(map(str, segment))
        f.write(f"线段 {idx + 1}: {segment_str}\n")

print(f"处理后的划分结果已保存到 {output_file}")

