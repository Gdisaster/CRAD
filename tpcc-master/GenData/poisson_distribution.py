from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 设置泊松分布的参数λ
lambda_val = 20

# 生成泊松分布的随机数
poisson_rvs = poisson.rvs(lambda_val,size=1000)
temp_list = []
for i in range(1000):
    temp_list.append(poisson.rvs(lambda_val))

count_list = sorted(Counter(poisson_rvs).items())# 将字典转换为键值对列表，并排序
print(count_list)

frequency_list = []
frequency_cnt = 0
for i in range(0, 22, 2):
    frequency_cnt = frequency_cnt + count_list[i][1]
    frequency_list.append(count_list[i][1])
    print(count_list[i][1], frequency_cnt)

for i in range(len(frequency_list)):
    frequency_list[i] = round(frequency_list[i] / frequency_cnt)
print(frequency_list)


# (11, 5), (12, 13), (13, 25), (14, 29), (15, 49), (16, 76), (17, 107), (18, 88), (19, 97), (20, 91), (21, 75), (22, 82)
# (10, 7), (11, 6), (12, 19), (13, 32), (14, 42), (15, 61), (16, 70), (17, 75), (18, 72), (19, 78), (20, 87), (21, 83)

# count_list = sorted(Counter(temp_list).items())# 将字典转换为键值对列表，并排序

# 使用zip提取x和y值
x, y = zip(*count_list)

# # 绘制图像
# plt.plot(x, y, marker='o')  # marker='o'表示用圆圈标记每个点

# 绘制条形图
plt.bar(x, y)

# 添加x轴和y轴的标签
plt.xlabel("Key")
plt.ylabel("Value")

# 添加图像的标题
plt.title("Dictionary as a Plot")

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