from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# 设置泊松分布的参数λ
lambda_val_1 = 5
lambda_val_2 = 35

# # 生成两个泊松分布的随机样本
# poisson_rvs_1 = poisson.rvs(lambda_val_1, size=1000)
# poisson_rvs_2 = poisson.rvs(lambda_val_2, size=1000)

# samples = np.concatenate((poisson_rvs_1, poisson_rvs_2))# 将两个样本合并，形成双峰泊松分布

# count_list = sorted(Counter(samples).items())# 将字典转换为键值对列表，并排序


temp_list = []
for i in range(2000):
    if i % 2 == 0:
        temp_list.append(poisson.rvs(lambda_val_2))
    else:
        temp_list.append(poisson.rvs(lambda_val_1))
count_list = sorted(Counter(temp_list).items())# 将字典转换为键值对列表，并排序

print(count_list)

# 使用zip提取x和y值
x, y = zip(*count_list)

# # 绘制图像
# plt.plot(x, y, marker='o')  # marker='o'表示用圆圈标记每个点

# 绘制条形图
plt.bar(x, y)

# 添加x轴和y轴的标签
plt.xlabel("Key")
plt.ylabel("Count")

# 添加图像的标题
plt.title("bimodal_poisson_distribution")

# 显示图像
plt.show()

# # 计算PMF
# k = np.arange(poisson_rvs.min(), poisson_rvs.max()+1)
# pmf = poisson.pmf(k, lambda_val)

# # 计算CDF
# cdf = poisson.cdf(k, lambda_val)

# # 绘制PMF
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.bar(k, pmf, width=0.5, alpha=0.5, color='blue', label='PMF')
# plt.title('Poisson Distribution PMF')
# plt.xlabel('k')
# plt.ylabel('P(X=k)')
# plt.legend()

# # 绘制CDF
# plt.subplot(1, 2, 2)
# plt.step(k, cdf, where='mid', label='CDF')
# plt.title('Poisson Distribution CDF')
# plt.xlabel('k')
# plt.ylabel('P(X<=k)')
# plt.legend()

# plt.tight_layout()
# plt.show()