# -*- coding: utf-8 -*-

def check_segments_grouping(txt_file, min_group_size=28):
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    segment_lengths = []
    for line in lines:
        if line.startswith("线段"):
            parts = line.strip().split(":")
            if len(parts) != 2:
                continue
            segment = parts[1].strip().split()
            segment = list(map(int, segment))
            segment_lengths.append((segment[0], segment[-1]))

    # 检查是否连续：例如，线段 1 的末尾是 100，线段 2 的起始是 101，则认为是连续的
    groups = []
    current_group = [segment_lengths[0]]
    for i in range(1, len(segment_lengths)):
        prev_end = segment_lengths[i - 1][1]
        curr_start = segment_lengths[i][0]

        if curr_start == prev_end + 1:
            current_group.append(segment_lengths[i])
        else:
            groups.append(current_group)
            current_group = [segment_lengths[i]]

    if current_group:
        groups.append(current_group)

    # 打印每组线段的信息
    print(f"共检测到 {len(groups)} 个连续线段组（半脑）")
    for i, group in enumerate(groups):
        print(f"组 {i+1} 包含 {len(group)} 条线段，起始点：{group[0][0]}，结束点：{group[-1][1]}")

if __name__ == "__main__":
    segments_txt_path = 'F:/Code/PythonProject/TransformerProject/New_projection/data/segments.txt'
    check_segments_grouping(segments_txt_path)
