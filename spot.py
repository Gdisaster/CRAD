import numpy as np
from scipy.stats import genpareto

class SPOT:
    def __init__(self, confidence_level, quantile=0.95):
        """
        初始化 SPOT 对象。
        参数:
            quantile (float): 初始阈值分位数，用于估算极值阈值。
            confidence_level (float): 置信水平。
        """
        self.quantile = quantile  # 分位数
        self.confidence_level = confidence_level # 置信水平
        self.t = None # 初始阈值
        self.threshold = None  # 动态阈值，对应算法中的zq
        self.gpd_params = None  # 广义帕累托分布（GPD）参数：形状参数 ξ, 位置参数 loc, 尺度参数 σ
        self.excesses = None # 超过阈值的数据集合，为了更新时拟合新的GPD使用

    def fit(self, data):
        """
        用于拟合历史数据，估算初始阈值和 GPD 参数。
        参数:
            data (np.array): 历史时间序列数据。
        """
        self.t = np.quantile(data, self.quantile)
        #print(f't:{self.t}')
        self.excesses = data[data > self.t] - self.t
        # print(f'excesses:{self.excesses+self.t}')
        if len(self.excesses) > 0:
            self.gpd_params = genpareto.fit(self.excesses)
            self.threshold = self.t + genpareto.ppf(self.confidence_level, *self.gpd_params)
            print(f'threshold:{self.threshold}')
        else:
            raise ValueError("没有超过阈值的点，无法拟合 GPD。")

    def detect(self, new_point):
        """
        检测新点是否为异常点。
        参数:
            new_point (float): 新时间序列点。
        返回:
            bool: 如果是异常点返回 True，否则返回 False。
        """
        if self.t is None or self.threshold is None or self.gpd_params is None:
            raise RuntimeError("请先调用 fit() 方法拟合历史数据。")

        if new_point > self.threshold:
            return True
        return False

    def update(self, new_point):
        """
        用新数据更新动态阈值和模型。
        异常值不会更新模型
        参数:
            new_point (float): 新时间序列点。
        """

        # print(f"更新前阈值: {self.threshold}")
        if new_point <= self.threshold and new_point > self.t:
            self.excesses = np.append(self.excesses, new_point - self.t)
            # print(f'excesses:{self.excesses+self.t}')
            self.gpd_params = genpareto.fit(self.excesses)
            self.threshold = self.t + genpareto.ppf(self.confidence_level, *self.gpd_params)
            
        # print(f"更新后阈值: {self.threshold}")
        # print()

# # 测试样例如下
# # 数据样本
# historical_data = np.array([1, 2, 3, 4, 5, 10, 15, 20, 25, 30])
# test_data = [3,31,29.5,30,27,29.99]

# # 初始化SPOT模型
# spot = SPOT(confidence_level=0.99)

# # 拟合历史数据
# spot.fit(historical_data)

# # 逐点检测
# for point in test_data:
#     is_anomaly = spot.detect(point)
#     print(f"检测点: {point}，其是否为异常点: {is_anomaly}")
#     # if is_anomaly:
#     #     print(f"异常点检测到: {point}")
#     # 更新模型
#     spot.update(point)