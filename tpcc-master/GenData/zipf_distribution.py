import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zipfian
from collections import Counter

# 设置Zipf分布的参数a和最大值N
a = 3
N = 8
zipf_dist = zipfian(a, N)# 创建Zipf分布对象
zipf_rvs = zipf_dist.rvs()# 生成单个Zipf分布的随机数

temp_list = []
for i in range(5000):
    temp_list.append(zipf_dist.rvs())
#count_list = sorted(Counter(temp_list).items())# 将字典转换为键值对列表，并排序
print(pd.value_counts(temp_list))

count_list = sorted(Counter(temp_list).items())

# 使用zip提取x和y值
x, y = zip(*count_list)

y = np.array(y)

# Min-Max 归一化
y = y / 5000
cnt = 0
for item in y:
    cnt += item
    print(item, cnt)

print(x,y)

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


# a = 2
# n = 9

# # 创建随机数生成器
# rng = np.random.default_rng()

# # # 生成Zipf分布的随机数并限制其范围
# # s = np.minimum(rng.zipf(a), n)

# list_a = []
# for i in range(10000):
#     list_a.append(np.minimum(rng.zipf(a), n))
# print(pd.value_counts(list_a))
