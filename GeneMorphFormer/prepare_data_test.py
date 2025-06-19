import numpy as np
import scipy.io
import os

# 初始化空矩阵，用来存储拼接后的结果
coord = []
curv = []
label = []

# 文件夹路径，包含你的.mat文件
mat_folder = 'F:/dataset/Extract_gyri_sulci_curv/Marmoset_extract_gene_feature_by_nissl_line' \
             '/Marmoset_Processed_Coords_and_Curvs_Smooth'  # 修改为实际路径

# 获取该文件夹下所有的.mat文件
mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]

# 遍历所有.mat文件
for mat_file in mat_files:
    # 加载当前.mat文件
    mat_path = os.path.join(mat_folder, mat_file)
    data = scipy.io.loadmat(mat_path)

    # 假设每个.mat文件中都有三个矩阵: matrix1, matrix2, matrix3
    if 'this_coord' in data and 'this_curv' in data and 'this_label' in data:
        matrix1 = data['this_coord']
        matrix2 = data['this_curv']
        matrix3 = data['this_label']

        # 将当前矩阵拼接到相应的大矩阵中
        coord.append(matrix1)
        curv.append(matrix2)
        label.append(matrix3)
    else:
        print(f"文件 {mat_file} 不包含 matrix1, matrix2 或 matrix3 矩阵")

# 拼接所有矩阵
coord = np.vstack(coord)  # 沿行拼接
curv = np.vstack(curv)  # 沿行拼接
label = np.vstack(label)  # 沿行拼接

# 保存拼接后的矩阵为一个新的.mat文件
scipy.io.savemat('F:/Code/PythonProject/TransformerProject/New_projection/data/all.mat', {'coord': coord, 'curv': curv, 'label': label})
