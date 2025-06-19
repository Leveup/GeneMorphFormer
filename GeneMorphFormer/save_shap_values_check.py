# -*- coding: utf-8 -*-
import json
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import h5py


# ----------------- 数据加载 -----------------
test_json = 'F:/Code/PythonProject/TransformerProject/New_projection/data/test_nor_0-600_test.json'
with open(test_json, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 筛选出没有增强的数据
original_data = [entry for entry in test_data if '_reversed' not in str(entry['segment_id'])]


# 提取对应的特征
def extract_features(data):
    return np.array([np.array([point[1] for point in item["data"]]) for item in data])


all_features = extract_features(original_data)
all_features_tensor = torch.tensor(all_features, dtype=torch.float32, requires_grad=True)
print("all_features_tensor shape", all_features_tensor.shape)

# ----------------- 加载 SHAP 值 -----------------
# 假设已经保存的 SHAP 值样本，加载指定的 SHAP 样本文件
shap_sample_index = 0  # 可以修改为你需要的样本索引，比如查看第0个样本的SHAP值
shap_filename = f"F:/Code/PythonProject/TransformerProject/New_projection/shap_values/shap_values_sample_{shap_sample_index}.npy"
all_shap_values_aggregated = np.load(shap_filename)
# 打印 SHAP 值的维度
print(f"Original SHAP values shape: {all_shap_values_aggregated.shape}")

# 聚合：对 (axis=1, axis=3) 维度进行求均值，转换成 (batch_size, 1024)  聚合维度 (1, 600, 1024, 600) -> (1, 1024)
shap_values_batch_aggregated = np.mean(all_shap_values_aggregated, axis=(1, 3))
print(f"Aggregated SHAP values shape: {shap_values_batch_aggregated.shape}")

# 聚合样本特征，取序列长度上的均值，得到形状 [1, 1024]
all_features_features_aggregated = all_features_tensor[shap_sample_index].mean(dim=0).unsqueeze(0).detach().cpu().numpy()
print(f"Aggregated all_features_tensor[shap_sample_index] shape: {all_features_features_aggregated.shape}")

# ----------------- 加载基因列表 -----------------
mat_file2 = 'F:/dataset/Extract_gyri_sulci_curv/Marmoset_extract_gene_feature_by_nissl_line' \
            '/Marmoset_Processed_Genes_feature/adjusted_expression_matrix.mat'
gene_list = []
with h5py.File(mat_file2, 'r') as f:
    for ref in f['gene_list']:
        raw_bytes = f[ref[0]][()].tobytes()
        gene_name = raw_bytes.decode('utf-16-le').strip('\x00')
        gene_list.append(gene_name)
print("Total genes:", len(gene_list))
print("Example genes:", gene_list[:5])

# ----------------- 可视化 分析基因重要性分布 -----------------
# 设置图形属性
plt.figure()
plt.gcf().set_size_inches(7, 6)
plt.rcParams['font.size'] = 12

# ----------------- 绘制 SHAP 重要性条形图 -----------------
plt.title(f"SHAP Value Analysis for Sample {shap_sample_index} - Mean(|SHAP Value|)")

# 绘制柱状图（即基因的 SHAP 重要性）
shap.summary_plot(shap_values_batch_aggregated, feature_names=gene_list, plot_type='bar', show=False)

# 修改柱状图颜色
ax = plt.gca()
for bar in ax.patches:
    bar.set_color("#ADD8E6")

# ----------------- 绘制散点图 -----------------
# 绘制散点图，显示各样本聚合特征对模型输出的影响
shap.summary_plot(
    shap_values_batch_aggregated,
    all_features_features_aggregated,  # 假设已经计算过聚合的特征
    feature_names=gene_list,
    cmap='Spectral',
    show=False
)

# 保存高曲率线段的可视化图像
plt.savefig(f'shap_sample_{shap_sample_index}_visualization.png', dpi=300, bbox_inches='tight')

# 输出信息
print(f"SHAP visualization for sample {shap_sample_index} saved.")
