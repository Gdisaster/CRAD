import pandas as pd
import torch
import math
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path
from ly_transformer import *
from spot import SPOT
from metrics import *
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import logging
import os
import sys

def generate_diagonal_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
):
    return torch.diag(torch.full((sz,), float('-inf'), dtype=dtype, device=device))

# 定义一个函数来转换单个字符串
def convert_time(time_str):
    # 去掉方括号和引号
    time_str = time_str.replace("[", "").replace("]", "").replace("'", "")
    # 分割日期和时间
    date_part, time_part = time_str.split(", ")
    # 拼接成所需格式
    formatted_time = date_part.replace("-", "") + time_part.replace(":", "")
    return int(formatted_time)

def extract_time_features(row_data, time_column='time_index1'):
    """
    从时间戳列中提取年、月、日、小时、分钟、秒，并将其转换为浮点格式。

    参数:
        row_data (pd.DataFrame): 包含时间戳列的 DataFrame。
        time_column (str): 时间戳列的名称，默认为 'time_index1'。

    返回:
        pd.DataFrame: 添加了年、月、日、小时、分钟、秒列的 DataFrame。
    """
    # 提取时间特征
    row_data['year'] = row_data[time_column] // 10000000000  # 提取年份
    row_data['month'] = (row_data[time_column] // 100000000) % 100  # 提取月份
    row_data['day'] = (row_data[time_column] // 1000000) % 100  # 提取日期
    row_data['hour'] = (row_data[time_column] // 10000) % 100  # 提取小时
    row_data['minute'] = (row_data[time_column] // 100) % 100  # 提取分钟
    row_data['second'] = row_data[time_column] % 100  # 提取秒

    # 将时间特征转换为浮点格式
    row_data['year'] = row_data['year'].apply(lambda x: np.float32(f"1.{x}"))
    row_data['month'] = row_data['month'].apply(lambda x: np.float32(f"1.{x:02d}"))
    row_data['day'] = row_data['day'].apply(lambda x: np.float32(f"1.{x:02d}"))
    row_data['hour'] = row_data['hour'].apply(lambda x: np.float32(f"1.{x:02d}"))
    row_data['minute'] = row_data['minute'].apply(lambda x: np.float32(f"1.{x:02d}"))
    row_data['second'] = row_data['second'].apply(lambda x: np.float32(f"1.{x:02d}"))

    return row_data

def get_time_info(time_int):
    Second = int(time_int % 100)
    Minute = int((time_int // 1e2) % 100)
    Hour = int(time_int // 1e4)
    Day = int(time_int // 1e6)
    Month = int((time_int // 1e8) % 100)

    date_part = int(time_int // 1e6)  # 整数除法去掉后 6 位，得到 20210806
    year = date_part // 10000          # 提取前 4 位作为年
    month = (date_part // 100) % 100   # 提取中间两位作为月
    day = date_part % 100              # 提取最后两位作为日 
    date_obj = datetime(year, month, day)
    _, Week, _ = date_obj.isocalendar()

    return Second, Minute, Hour, Day, Month, Week

# 定义滑动窗口数据集
class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size):
        self.X, self.Y = self.create_sliding_window(data, window_size)
        # self.X, self.Y, self.label_truth = self.create_sliding_window(data, window_size)

    @staticmethod 
    def create_sliding_window(data, window_size):
        X, Y = [], []
        # X, Y, label_truth = [], [], []
        
        ### Version 3
        for i in range(len(data) - window_size + 1):
            X.append(data[i:i + window_size, :])
            Y.append(data[i:i + window_size, :])
            # label_truth.append(data[i:i + window_size, 0])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
        # return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), np.array(label_truth, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).to(device), torch.from_numpy(self.Y[idx]).to(device)
        # return torch.from_numpy(self.X[idx]).to(device), torch.from_numpy(self.Y[idx]).to(device), torch.from_numpy(self.label_truth[idx]).to(device)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :] #相当于只截取前seq_len个位置编码
        return self.dropout(x)
    
class myTransformer(nn.Module):
    def __init__(self):
        super(myTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model = d_model, nhead= n_heads,
                                          num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                          dim_feedforward=dim_feedforward)
        
        encoder_layers = TransformerEncoderLayer(d_model= d_model, nhead=n_heads, dim_feedforward= dim_feedforward, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        decoder_layer = TransformerDecoderLayer(d_model= d_model, nhead= n_heads, dim_feedforward= dim_feedforward, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layer, n_layers)

        self.src_emb = nn.Linear(input_dim, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.tgt_emb = nn.Linear(input_dim, d_model)
        self.fc = nn.Linear(d_model, input_dim)  # 输出维度回到特征数

        self.conv1d = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)  # 卷积核尺寸为 1x4
        # self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, int((input_dim + d_model) / 2)),  # 输入维度为 input_dim，隐藏层维度为 (input_dim + d_model) / 2
            nn.ReLU(),                 # 激活函数
            nn.Linear(int((input_dim + d_model) / 2), d_model)  # 输出维度变为 d_model
        )

        self.mlp_time = nn.Sequential(
            nn.Linear(6, 32),  # 6 → 32
            nn.ReLU(),
            nn.Linear(32, 64),  # 32 → 64
            nn.ReLU(),
            nn.Linear(64, 128),  # 64 → 128
            nn.ReLU(),
            nn.Linear(128, 256),  # 128 → 256
            nn.ReLU(),
            nn.Linear(256, 512),  # 256 → 512
        )
        self.relu = nn.ReLU()

        self.mlp_output = nn.Sequential(
            nn.Linear(d_model, int((input_dim + d_model) / 2)),  # 输入维度为 input_dim，隐藏层维度为 (input_dim + d_model) / 2
            nn.ReLU(),                 # 激活函数
            nn.Linear(int((input_dim + d_model) / 2), input_dim)  # 输出维度变为 d_model
        )

    def forward(self, enc_inputs, dec_inputs, mode='train'):
        """
        enc_inputs: [batch_size, window_size, input_dim]  # 前99条数据
        # dec_inputs: [batch_size, window_size, input_dim]  # 当前目标sql
        dec_inputs: [batch_size, input_dim]  # 当前目标sql(填0)
        """

        batch_size = enc_inputs.shape[0]
        window_size = enc_inputs.shape[1]
        input_dim = enc_inputs.shape[2]

        device = enc_inputs.device

        # 初始化一个与 enc_inputs 同尺寸的全零张量，并将其放置在相同的设备上
        tensor_time = torch.zeros([batch_size, window_size, 512], device=device)

        # # 遍历第一个维度batch（32）
        # for i in range(batch_size):
        #     # 遍历第二个维度window（100）
        #     for j in range(window_size):
                
        #         # 12021.0, 108.0, 106.0
        #         year, month, day = torch.round(enc_inputs[i][j][input_dim - 6] * 1e4), torch.round(enc_inputs[i][j][input_dim - 5] * 1e2), torch.round(enc_inputs[i][j][input_dim - 4] * 1e2)
        #         # 106.0, 133.0, 100.0
        #         hour, minute, second = torch.round(enc_inputs[i][j][input_dim - 3] * 1e2), torch.round(enc_inputs[i][j][input_dim - 2] * 1e2), torch.round(enc_inputs[i][j][input_dim - 1] * 1e2)
        #         # Second, Minute, Hour, Day, Month, Week = get_time_info(enc_inputs[i][j][0])

        #         year_i = (int(year % 1e4))
        #         month_i = (int(month % 1e2))
        #         day_i = (int(day % 1e2))
        #         hour_i = (int(hour % 1e2))
        #         minute_i = (int(minute % 1e2))
        #         second_i = (int(second % 1e2))
        #         list_time = [year_i, month_i, day_i, hour_i, minute_i, second_i]

        #         # 将列表转换为张量
        #         input_tensor = torch.tensor(list_time, dtype=torch.float32, device=device)

        #         tensor_time[i][j] = self.mlp_time(input_tensor)
                
        #         Hour = str(int(year))[1:] + str(int(month))[1:] + str(int(day))[1:] + str(int(hour))[1:]
        #         Day = str(int(year))[1:] + str(int(month))[1:] + str(int(day))[1:]
        #         Month = str(month_i)

        #         date_obj = datetime(year_i, month_i, day_i)
        #         _, Week, _ = date_obj.isocalendar()

        #         if mode == 'train':
        #             vector_hour = hour_vectors[Hour]
        #             vector_day = day_vectors[Day]
        #             vector_month = month_vectors[Month]
        #             vector_week = week_vectors[str(Week)]
        #         else:
        #             vector_hour = test_hour_vectors[Hour]
        #             vector_day = test_day_vectors[Day]
        #             vector_month = test_month_vectors[Month]
        #             vector_week = test_week_vectors[str(Week)]
                
        #         vector_hour_tensor = torch.tensor(vector_hour, device=device).squeeze().clone().detach()
        #         vector_day_tensor = torch.tensor(vector_day, device=device).squeeze().clone().detach()
        #         vector_month_tensor = torch.tensor(vector_month, device=device).squeeze().clone().detach()
        #         vector_week_tensor = torch.tensor(vector_week, device=device).squeeze().clone().detach()
                                
        #         matrix_4 = torch.stack([vector_hour_tensor, vector_day_tensor, vector_month_tensor, vector_week_tensor], dim=0).to(torch.float32).to(device)
        #         output_4 = self.conv1d(matrix_4.unsqueeze(0)).squeeze(0).squeeze(0)
        #         # print(f"matrix_4: {matrix_4.shape}, output_4: {output_4.shape}")
        #         # sys.exit()

        #         enc_inputs[i, j, :-6] += self.relu(output_4)

                # print(f"time_index1: {enc_inputs[i][j][0]}")
                # sys.exit()

        enc_inputs = enc_inputs[:, :, :-6]
        # dec_inputs = dec_inputs[:, :-6].transpose(1, 2)

        enc_inputs = self.relu(enc_inputs)
        dec_inputs = self.relu(dec_inputs)

        batch_size, src_len, input_dim = enc_inputs.shape
        batch_size, tgt_len, input_dim = dec_inputs.shape

        # 1. 嵌入映射
        # enc_outputs = self.src_emb(enc_inputs)  # [batch_size, window_size, d_model]
        # dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, window_size, d_model]

        enc_outputs = self.mlp(enc_inputs)  # [batch_size, window_size, d_model]
        dec_outputs = self.mlp(dec_inputs)  # [batch_size, window_size, d_model]

        # 2. 添加位置编码
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1))  # [window_size, batch_size, d_model] [100, 32, 512]
        # dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1))  # [window_size, batch_size, d_model] [1, 32, 512]
        dec_outputs = dec_outputs.transpose(0, 1) # [window_size, batch_size, d_model] [1, 32, 512]
        
        # 添加全局时间编码
        enc_outputs += tensor_time.transpose(0, 1)


        # 3. Key Padding Mask
        # 假设输入中的 padding 值为0，生成 mask
        src_key_padding_mask = enc_inputs.sum(dim=-1).eq(0).to(torch.bool)  # [batch_size, window_size]
        tgt_key_padding_mask = dec_inputs.sum(dim=-1).eq(0).to(torch.bool)  # [batch_size, window_size]

        # 4. Mask for decoder (用于屏蔽未来位置)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(enc_inputs.device).to(torch.bool)

        # 5. Transformer 编码和解码
        # output = self.transformer(
        #     src=enc_outputs,
        #     tgt=dec_outputs,
        #     src_key_padding_mask=src_key_padding_mask,
        #     # tgt_key_padding_mask=tgt_key_padding_mask,
        #     tgt_mask=tgt_mask
        # )  # [window_size, batch_size, d_model]
            
        op_encoder = self.transformer_encoder(src=enc_outputs, src_key_padding_mask = src_key_padding_mask)
        output = self.transformer_decoder(tgt = dec_outputs, memory = op_encoder)  #tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask
        
        # 6. 提取最后一层输出并映射到输入维度
        output = self.mlp_output(output)  # [window_size, input_dim]: [1, 32, 512] → [32, 512] → [32, 199]

        # print(f"output: {output.shape}")
        # sys.exit()

        return output.transpose(0, 1)



# 检查是否可用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# if torch.cuda.is_available():
#     torch.cuda.set_device(1)
#     device = torch.device("cuda:1")
# else:
#     device = 'cpu'
print(f"device: {device}")

role = "1"
anomaly = '-a22'
train_percent = '-20%'
test_percent = '-20%'

spot = SPOT(confidence_level=0.9, quantile=0.85)

# '训练'数据加载
file_path = "./data/Data_tpcc_ly/DataNoTimeFusion/"
file_row = role + test_percent + ".csv"  #"0-80%.csv"
row_data = pd.read_csv(file_path + file_row)
# '测试'数据加载
test_row = role + test_percent + anomaly + ".csv" #"0-20%-a11.csv"
test_data = pd.read_csv(file_path + test_row)

print(f"row_data: {file_path + file_row}, test_data: {file_path + test_row}")
# 多尺度'训练'数据加载
path_hour = Path('./data/Data_tpcc_ly/DataWithTimeFusionHour/HourTimeFusion-' + role + train_percent)
path_day = Path('./data/Data_tpcc_ly/DataWithTimeFusionDay/DayTimeFusion-' + role + train_percent)
path_month = Path('./data/Data_tpcc_ly/DataWithTimeFusionMonth/MonthTimeFusion-' + role + train_percent)
path_week = Path('./data/Data_tpcc_ly/DataWithTimeFusionWeek/WeekTimeFusion-' + role + train_percent)
# 多尺度'测试'数据加载
test_path_hour  = Path('./data/Data_tpcc_ly/DataWithTimeFusionHour/HourTimeFusion-' + role + test_percent + anomaly)
test_path_day   = Path('./data/Data_tpcc_ly/DataWithTimeFusionDay/DayTimeFusion-' + role + test_percent + anomaly)
test_path_month = Path('./data/Data_tpcc_ly/DataWithTimeFusionMonth/MonthTimeFusion-' + role + test_percent + anomaly)
test_path_week  = Path('./data/Data_tpcc_ly/DataWithTimeFusionWeek/WeekTimeFusion-' + role + test_percent + anomaly)

# 归一化
scaler = MinMaxScaler()

# 初始化日志文件
log_file = "./Log/" + role + "/result_log_" + role + anomaly + ".log"

# 确保日志路径存在
if os.path.exists(log_file):
    os.remove(log_file)
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# 配置日志记录
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # 时间、日志级别、信息
    datefmt="%Y-%m-%d %H:%M:%S"  # 日期格式
)

hour_vectors = {}
day_vectors = {}
month_vectors = {}
week_vectors = {}

test_hour_vectors = {}
test_day_vectors = {}
test_month_vectors = {}
test_week_vectors = {}

print(f"row_data: {row_data.iloc[0]}")

row_data['time_index1'] = row_data['time_index1'].apply(convert_time)
test_data['time_index1'] = test_data['time_index1'].apply(convert_time)

row_data = extract_time_features(row_data)
test_data = extract_time_features(test_data)

label_test_groundtruth = test_data['label']

row_data = row_data.drop(columns=['time_index1', 'time_index2', 'label', 'sql_id'])
test_data = test_data.drop(columns=['time_index1', 'time_index2', 'label', 'sql_id'])

row_data_normalize = row_data.iloc[:, :-6]
test_data_normalize = test_data.iloc[:, :-6]

row_data.iloc[:, :-6] = scaler.fit_transform(row_data_normalize)
test_data.iloc[:, :-6] = scaler.transform(test_data_normalize)

### 读取多尺度hour数据
for csv_file in path_hour.glob('*.csv'):
    hour_key = csv_file.name[:-4]
    hour_vectors[hour_key] = scaler.transform(pd.read_csv(csv_file).iloc[:, 1:])

for test_csv_file in test_path_hour.glob('*.csv'):
    test_hour_key = test_csv_file.name[:-4]
    test_hour_vectors[test_hour_key] = scaler.transform(pd.read_csv(test_csv_file).iloc[:, 1:])

### 读取多尺度day数据
for csv_file in path_day.glob('*.csv'):
    day_key = csv_file.name[:-4]
    day_vectors[day_key] = scaler.transform(pd.read_csv(csv_file).iloc[:, 1:])

for test_csv_file in test_path_day.glob('*.csv'):
    test_day_key = test_csv_file.name[:-4]
    test_day_vectors[test_day_key] = scaler.transform(pd.read_csv(test_csv_file).iloc[:, 1:])

### 读取多尺度month数据
for csv_file in path_month.glob('*.csv'):
    month_key = csv_file.name[:-4]
    month_vectors[month_key] = scaler.transform(pd.read_csv(csv_file).iloc[:, 1:])

for test_csv_file in test_path_month.glob('*.csv'):
    test_month_key = test_csv_file.name[:-4]
    test_month_vectors[test_month_key] = scaler.transform(pd.read_csv(test_csv_file).iloc[:, 1:])

### 读取多尺度week数据
for csv_file in path_week.glob('*.csv'):
    week_key = csv_file.name[:-4]
    week_vectors[week_key] = scaler.transform(pd.read_csv(csv_file).iloc[:, 1:])

for test_csv_file in test_path_week.glob('*.csv'):
    test_week_key = test_csv_file.name[:-4]
    test_week_vectors[test_week_key] = scaler.transform(pd.read_csv(test_csv_file).iloc[:, 1:])

# 创建数据集
window_size = 100
batch_size = 64
row_dataset = SlidingWindowDataset(row_data.values, window_size)
row_loader = DataLoader(row_dataset, batch_size, shuffle=False)
# print(f"row_dataset: {row_dataset.X.shape, row_dataset.Y.shape}")

test_dataset = SlidingWindowDataset(test_data.values, window_size)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
# print(f"test data: {test_data.shape, label_test_groundtruth.shape}")

# 模型初始化
input_dim = row_dataset.X.shape[2] - 6
output_dim = input_dim - 1
hidden_dim = 1024
d_model = 512
n_heads = 8
n_layers = 3
dim_feedforward = 1024

model = myTransformer().cuda()
criterion = nn.MSELoss()  # 回归任务
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = torch.optim.AdamW(model.parameters() , lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

# 训练循环
epochs = 200

# spot
flag_spot = True

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, (x, y) in enumerate(row_loader):
        ### x: torch.Size([32, 100, 205]), y: torch.Size([32, 100, 199]), label_truth: torch.Size([32, 100])
        # x, y, label_truth = x.to(device), y.to(device), label_truth.to(device)
        # print(f"x: {x.shape}, y: {y.shape}")

        x, y = x.to(device), y.to(device)
        
        y = y[:, :, :-6]
        zero_y = torch.zeros_like(y)
        
        ### output: torch.Size([32, 100, 199])
        # print(f"output: {output.shape}")

        output = model(x, zero_y, mode = 'train')  # 输出预测值
        
        mse_loss = criterion(output, y)  # 计算损失
        cos_sim = F.cosine_similarity(output, y, dim=-1)
        cos_loss = torch.mean(1 - cos_sim)

        loss = mse_loss * 0.9 + cos_loss * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # logging.info(
        #     f"train: epoch: {epoch + 1} / {epochs}, step: {step}  / {len(row_loader)}, loss: {loss.item()}, mse_loss: {mse_loss}, cos_loss: {cos_loss}"
        # )
        print(f"time: {datetime.now()}, train: epoch: {epoch + 1} / {epochs}, step: {step}  / {len(row_loader)}, mse_loss: {mse_loss.item()}, cos_loss: {cos_loss.item()}, loss: {loss.item()}")

    scheduler.step()

    logging.info(f"time: {datetime.now()}, train: Epoch {epoch + 1}, loader: {len(row_loader)}, total_loss: {total_loss / len(row_loader):.6f}")
    # print(f"train: Epoch {epoch + 1}, loader: {len(row_loader)}, total_loss: {total_loss / len(row_loader):.6f}")

    # 保存模型参数和优化器状态
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, 'checkpoint' + role + anomaly + '.pth')
    # print("模型参数和优化器状态已保存到 'checkpoint.pth'")

    if (epoch + 1) % 5 == 0:
        if flag_spot:
            flag_spot = False

            mse_loss = torch.mean((output - y) ** 2, dim=-1)
            cos_loss = 1 - F.cosine_similarity(output, y, dim=-1)
            loss = mse_loss * 0.9 + cos_loss * 0.1

            loss = loss.flatten().cpu().detach().numpy()
            spot.fit(loss)

        model.eval()
        with torch.no_grad():
            label_dict = {i: [] for i in range(label_test_groundtruth.shape[0])}
            for step, (x, y) in enumerate(test_loader):
                            
                x, y = x.to(device), y.to(device)
                y = y[:, :, :-6]
                zero_y = torch.zeros_like(y)
                output = model(x, zero_y, mode = 'test')  # 输出预测值

                # 对第3维（199维）求均值，得到32个batch，每个batch有100个数据, [0, 99][1, 100]...[31, 130]  [32, 131]...
                mse_loss = torch.mean((output - y) ** 2, dim=-1)  ##[32, 100]
                cos_loss = 1 - F.cosine_similarity(output, y, dim=-1)  ##[32, 100]
                loss = mse_loss * 0.9 + cos_loss * 0.1

                index = step * batch_size
                for i in range(loss.shape[0]):
                    for j in range(loss.shape[1]):
                        label_dict[int(index + i + j)].append(loss[i][j].cpu().numpy())
                        # logging.info(f"index: {int(index + i + j)}")
                        # if i == 0 and j == 0 or i == loss.shape[0] - 1 and j == loss.shape[1] - 1:
                        #     print(f"test step: {step}, x: {x.shape}, index: {int(index + i + j)}")

            label_pred = []
            mean_label_dict = {
                key: sum(value) / len(value) if len(value) > 0 else 0
                for key, value in label_dict.items()
            }
            value_label_dict = [value for value in mean_label_dict.values()]
            # print(f"value_label_dict: {len(value_label_dict)}")
            # loss = loss.flatten().cpu().numpy()
            for point in value_label_dict:
                is_anomaly = spot.detect(point)
                label_pred.append(1 if is_anomaly else 0)
                spot.update(point)

            f1 = f1_score(y_true = label_test_groundtruth, y_pred = label_pred, average='binary')  # 二分类任务

            TP, TN, FP, FN, TP_f, FN_f = calculate_confusion_matrix(y_true = label_test_groundtruth, y_pred = label_pred)
            metrics = calculate_metrics(TP, TN, FP, FN)

            # logging.info(
            #     f"test: epoch: {epoch + 1} / {epochs}, step: {step}  / {len(test_loader)},  TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']},  FPR: {metrics['FPR']:.4f}, FNR: {metrics['FNR']:.4f}, Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, F1: {metrics['F1']:.4f}"
            # )

            logging.info(
                f"time: {datetime.now()}, test: epoch: {epoch + 1} / {epochs}, step: {step}  / {len(test_loader)}, "
                f"TP: {metrics['TP'], TP_f}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN'], FN_f}, "
                f"FPR: {metrics['FPR']:.4f}, FNR: {metrics['FNR']:.4f}, "
                f"Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, F1: {metrics['F1']:.4f}"
            )
            print(f"time: {datetime.now()}, test: epoch: {epoch + 1} / {epochs}, step: {step}  / {len(test_loader)}, f1: {f1}")

# 保存模型参数和优化器状态
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict()
}
torch.save(checkpoint, 'checkpoint' + role + anomaly + '.pth')
# print("模型参数和优化器状态已保存到 'checkpoint.pth'")

# # 加载模型参数和优化器状态
# model = myTransformer().cuda()
# optimizer = torch.optim.AdamW(model.parameters() , lr=1e-4, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

# checkpoint = torch.load('checkpoint.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

model.eval()
with torch.no_grad():
    
    loss_train = {i: [] for i in range(row_data.shape[0])}
    loss_test = {i: [] for i in range(label_test_groundtruth.shape[0])}

    for step, (x, y) in enumerate(row_loader):

        x, y = x.to(device), y.to(device)
        y = y[:, :, :-6]
        zero_y = torch.zeros_like(y)
        output = model(x, zero_y, mode = 'test')  # 输出预测值
        
        mse_loss = torch.mean((output - y) ** 2, dim=-1)  ##[32, 100]
        cos_loss = 1 - F.cosine_similarity(output, y, dim=-1)  ##[32, 100]
        loss = mse_loss * 0.9 + cos_loss * 0.1

        index = step * batch_size
        for i in range(loss.shape[0]):
            for j in range(loss.shape[1]):
                loss_train[int(index + i + j)].append(loss[i][j].cpu().numpy())

    for step, (x, y) in enumerate(test_loader):
                    
        x, y = x.to(device), y.to(device)
        y = y[:, :, :-6]
        zero_y = torch.zeros_like(y)
        output = model(x, zero_y, mode = 'test')  # 输出预测值

        # 对第3维（199维）求均值，得到32个batch，每个batch有100个数据, [0, 99][1, 100]...[31, 130]  [32, 131]...
        mse_loss = torch.mean((output - y) ** 2, dim=-1)  ##[32, 100]
        cos_loss = 1 - F.cosine_similarity(output, y, dim=-1)  ##[32, 100]
        loss = mse_loss * 0.9 + cos_loss * 0.1

        index = step * batch_size
        for i in range(loss.shape[0]):
            for j in range(loss.shape[1]):
                loss_test[int(index + i + j)].append(loss[i][j].cpu().numpy())

    label_test = []

    mean_loss_train = {
        key: sum(value) / len(value) if len(value) > 0 else 0
        for key, value in loss_train.items()
    }
    value_loss_train = [value for value in mean_loss_train.values()]

    mean_loss_test = {
        key: sum(value) / len(value) if len(value) > 0 else 0
        for key, value in loss_test.items()
    }
    value_loss_test = [value for value in mean_loss_test.values()]
    # print(f"value_label_dict: {len(value_label_dict)}")
    # loss = loss.flatten().cpu().numpy()

    spot.fit(value_loss_train)

    for point in value_loss_test:
        is_anomaly = spot.detect(point)
        label_test.append(1 if is_anomaly else 0)
        spot.update(point)

    f1 = f1_score(y_true = label_test_groundtruth, y_pred = label_test, average='binary')  # 二分类任务

    TP, TN, FP, FN, TP_f, FN_f = calculate_confusion_matrix(y_true = label_test_groundtruth, y_pred = label_test)
    metrics = calculate_metrics(TP, TN, FP, FN)

    # logging.info(
    #     f"test: epoch: {epoch + 1} / {epochs}, step: {step}  / {len(test_loader)},  TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']},  FPR: {metrics['FPR']:.4f}, FNR: {metrics['FNR']:.4f}, Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, F1: {metrics['F1']:.4f}"
    # )

    logging.info(
        f"time: {datetime.now()}, test: epoch: {epoch + 1} / {epochs}, step: {step}  / {len(test_loader)}, "
        f"TP: {metrics['TP'], TP_f}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN'], FN_f}, "
        f"FPR: {metrics['FPR']:.4f}, FNR: {metrics['FNR']:.4f}, "
        f"Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, F1: {metrics['F1']:.4f}"
    )
    print(f"time: {datetime.now()}, test: epoch: {epoch + 1} / {epochs}, step: {step}  / {len(test_loader)}, f1: {f1}")