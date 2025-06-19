import matplotlib.pyplot as plt
import numpy as np

# 五折交叉验证的评估指标
folds = [1, 2, 3, 4, 5]
mse_list = [0.5822, 0.4038, 1.0307, 0.3610, 0.3169]
mae_list = [0.3570, 0.2967, 0.3803, 0.3118, 0.2855]
hd_list  = [0.9352, 0.7865, 1.0042, 0.8063, 0.7399]

# 创建绘图
plt.figure(figsize=(8, 6))

# 绘制三条曲线
plt.plot(folds, mse_list, marker='o', linestyle='-', color='blue', label='MSE')
plt.plot(folds, mae_list, marker='s', linestyle='--', color='orange', label='MAE')
plt.plot(folds, hd_list,  marker='^', linestyle='-.', color='green', label='HD')

# 图形设置
plt.xlabel("Fold")
plt.ylabel("Metric Value")
plt.title("Five-Fold Cross-Validation Results")
plt.xticks(folds)
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存为300dpi图像
plt.savefig("five_fold_metrics_combined.png", dpi=300)
plt.show()
