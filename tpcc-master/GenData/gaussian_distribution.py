import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def gaussian_tpcc(mean, std_dev):
    continuous_gaussian = np.random.normal(mean, std_dev)

    # 将连续值四舍五入到最近的整数
    discrete_value = np.round(continuous_gaussian)

    return int(discrete_value)

def discrete_gaussian(mean, std_dev, min_val, max_val):
    # """
    # 生成离散高斯分布的随机数。
    
    # 参数:
    # mean -- 离散高斯分布的均值
    # std_dev -- 离散高斯分布的标准差
    # min_val -- 随机数的最小值
    # max_val -- 随机数的最大值
    
    # 返回:
    # 一个离散高斯分布的随机整数
    # """
    # 连续高斯分布
    continuous_gaussian = np.random.normal(mean, std_dev)
    
    # 将连续值四舍五入到最近的整数
    discrete_value = np.round(continuous_gaussian)
    
    # 将值限制在[min_val, max_val]范围内
    # discrete_value = np.clip(discrete_value, min_val, max_val)
    
    return int(discrete_value)

# 示例：生成均值为0，标准差为1的离散高斯分布随机数，范围在-10到10之间
mean = 0
std_dev = 10
min_val = -30
max_val = 30
# random_number = discrete_gaussian(mean, std_dev, min_val, max_val)

list_a = []
for i in range(10000):
    list_a.append(discrete_gaussian(mean, std_dev, min_val, max_val))
# print(pd.value_counts(list_a))
    # random_number = discrete_gaussian(mean, std_dev, min_val, max_val)
    # print(random_number)

count_list = sorted(Counter(list_a).items())# 将字典转换为键值对列表，并排序

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
