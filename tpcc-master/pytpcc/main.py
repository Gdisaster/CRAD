
import numpy as np
from multiprocessing.pool import *
from scipy.stats import poisson
from collections import Counter
import datetime
import pandas as pd
import random
import psycopg2
from util.LY_SQL import *
from util.constants import *
from scipy.stats import zipfian
from util.markov_sql import *
from util.character_sql import Characteristics
from util.set_value import set_value

config = {
        "host":         ("The hostname to postgresql", "localhost" ),
        "port":         ("The port number to postgresql", 5432 ),
        "dbname":       ("Database name", "postgres"),
        "user":         ("user of the database", "postgres"),
        "password":     ("the password", "postgres")
    }

duration = 10
client = 3
const_week = 5
time_morning = ["6:00", "6:30", "7:00", "7:30", "8:00", "8:30", "9:00", "9:30", "10:00", "10:30", "11:00", "11:30"]
time_afternoon = ["13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30", "17:00", "17:30", "18:00", "18:30",
                   "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30"]
# sql_per_day = 300
# week_cnt = 1
# day_per_week = 5
# sql = "insert into a values('3');"

def gaussian_tpcc(mean, std_dev):
    continuous_gaussian = np.random.normal(mean, std_dev)

    # 将连续值四舍五入到最近的整数
    discrete_value = np.round(continuous_gaussian)

    return int(discrete_value)

def get_first_sql(user_id):

    user = Users[user_id]

    table_a = zipF_table[user]["a"]
    table_n = zipF_table[user]["N"]
    table_index = zipfian(table_a, table_n).rvs()

    if table_index >= table_n - 1:
        table_index = table_n - 1

    table = Tables[user][table_index]

    sql_a = zipF_sql[user][table]["a"]
    sql_n = zipF_sql[user][table]["N"]
    sql_index = zipfian(sql_a, sql_n).rvs()
    # print(f"sql_index: {sql_index}, sql_index type:{type(sql_index)}")

    sql = SQLS_user_table_id[user][table][sql_index]
    characteristics = Characteristics[user][table][sql_index]
    value = []
    for characteristic in characteristics[2]:
        value.append(set_value(characteristics[0], characteristic))

    sql_id = 0 #1...41...75(41 + 34)...114(41 + 34 + 39)
    if user_id == 1:  #添加在自己之前的角色下的sql数
        sql_id += 41
    if user_id == 2:
        sql_id += 75

    for table_id in range(table_index):
        table_name = Tables[user][table_id]
        sql_id += len(SQLS_user_table_id[user][table_name]) #添加在这个表之前的其他表下的sql数

    sql_id += sql_index #添加这个sql在这个表的id

    # sql = sql[0] % tuple(value)
    # print(sql)
    return table, sql[0], characteristics, tuple(value), sql_id

def get_sql_markov(user_id, sql_id, iteration):
    user = Users[user_id]
    table = ''
    sql_id -= 1

    # 1...41...75(41 + 34)...114(41 + 34 + 39)
    if user_id == 1:
        sql_id -= 41
    if user_id == 2:
        sql_id -= 75

    pro_markov = np.dot(np.array(p_markov[user][sql_id]), np.array(p_markov[user]) ** iteration)

    random_number = random.random() * sum(pro_markov)
    # 初始化累加和和计数器
    cumulative_sum = 0
    counter = 0
    # 遍历列表，累加元素值，直到累加和超过随机数
    for num in pro_markov:
        cumulative_sum += num
        counter += 1
        if cumulative_sum >= random_number:
            break
    new_sql_id = sql_index = counter
    print(f"new_sql_id: {new_sql_id}")
    for table_id in range(len(Tables[user])):
        if sql_index > len(SQLS_user_table_id[user][Tables[user][table_id]]):
            sql_index -= len(SQLS_user_table_id[user][Tables[user][table_id]])
        else:
            table = Tables[user][table_id]
            break

    sql = SQLS_user_table_id[user][table][sql_index]
    characteristics = Characteristics[user][table][sql_index]
    value = []
    for characteristic in characteristics[2]:
        value.append(set_value(characteristics[0], characteristic))

    if user_id == 1:  #添加在自己之前的角色下的sql数
        new_sql_id += 41
    if user_id == 2:
        new_sql_id += 75

    return table, sql[0], characteristics, tuple(value), new_sql_id

def get_frequency_morning():
    # 设置泊松分布的参数λ
    lambda_val = 20

    # 生成泊松分布的随机数
    poisson_rvs = poisson.rvs(lambda_val, size=1000)
    # temp_list = []
    # for i in range(1000):
    #     temp_list.append(poisson.rvs(lambda_val))

    count_list = sorted(Counter(poisson_rvs).items())  # 将字典转换为键值对列表，并排序
    # print(count_list)

    frequency_list = []
    frequency_cnt = 0
    for i in range(0, 23, 2):
        frequency_cnt = frequency_cnt + count_list[i][1]
        frequency_list.append(count_list[i][1])
        # print(count_list[i][1], frequency_cnt)

    for i in range(len(frequency_list)):
        frequency_list[i] = frequency_list[i] / frequency_cnt
    # print(frequency_list)

    return frequency_list

def get_frequency_afternoon():
    lambda_val_1 = 5
    lambda_val_2 = 35
    temp_list = []
    for i in range(2000):
        if i % 2 == 0:
            temp_list.append(poisson.rvs(lambda_val_2))
        else:
            temp_list.append(poisson.rvs(lambda_val_1))
    count_list = sorted(Counter(temp_list).items())

    frequency_list = []
    frequency_cnt = 0
    for i in range(0, 45, 2):
        frequency_cnt = frequency_cnt + count_list[i][1]
        frequency_list.append(count_list[i][1])
        # print(count_list[i][1], frequency_cnt)

    for i in range(len(frequency_list)):
        frequency_list[i] = frequency_list[i] / frequency_cnt

    return frequency_list

def execute_sql(user_id):

    conn = psycopg2.connect(database=config["dbname"][1], user=config["user"][1], password=config["password"][1],
                            host=config["host"][1], port=config["port"][1])
    cursor = conn.cursor()

    time = ''
    date_time = "2020-01-01"
    date_obj = datetime.datetime.strptime(date_time, "%Y-%m-%d")
    days_to_add = 0

    columns = ['time_index1', 'time_index2', 'sql_id', 'insert', 'delete', 'update', 'select']

    outer_keys = list(columns_tables.keys())

    # 提取每个子字典的键并存到列表中
    inner_keys = []
    for table_name in columns_tables:
        outer_keys.append(table_name + "-selectivity")
        inner_keys.extend(columns_tables[table_name].keys())

    # 拼接三个列表
    columns = columns + outer_keys + inner_keys

    data = []

    for week_cnt in range(1, const_week):  #week_cnt: 周数
        sql_per_day = 300  #sql_per_day： 每天执行sql基数

        for day in range(1, 8):

            delta = datetime.timedelta(days = days_to_add)
            new_date_obj = date_obj + delta
            date_time = new_date_obj.strftime("%Y-%m-%d")
            days_to_add += 1

            if week_cnt % 2 == 1 and (day == 6 or day == 7):
                continue

            if week_cnt % 2 == 0 and (day == 7):
                continue

            if week_cnt % 2 == 0 and (day == 6):
                sql_per_day = 0.8 * sql_per_day

            sql_gauss = gaussian_tpcc(0, 10)
            sql_num = sql_per_day + sql_gauss  #sql_num: 每天真正执行sql数

            frequency_morning = get_frequency_morning()
            frequency_afternoon = get_frequency_afternoon()
            frequency_day = frequency_morning + frequency_afternoon

            sql_day = [round(fre * sql_num * 0.5) for fre in frequency_day]

            time_day = time_morning + time_afternoon

            # print(f"sql_day:{sql_day}")
            # print(f"sql_day:{sql_day}, time_day:{time_day}")

            for time_period in range(len(time_day)): #每天的时间段

                sqls_period = sql_day[time_period]
                initial_time = time_day[time_period]

                # sql_id = 0
                time_last = ''
                time_last_timetype = datetime.datetime.strptime(time_last, "%H:%M:%S")
                for exe_sql_num in range(sqls_period): #时间段time_period内执行sql数
                    time_index1 = []
                    time_index2 = []
                    sql_character = [0 for _ in range(len(columns))]
                    
                    if exe_sql_num == 0:
                        table, sql, characteristic, value, sql_id = get_first_sql(user_id)

                        origin_time = datetime.datetime.strptime(initial_time, "%H:%M").replace(second=0).time()

                        if origin_time.hour == time_last_timetype.hour and origin_time.minute < time_last_timetype.minute:
                            new_time = datetime.datetime.combine(datetime.date.today(), origin_time) + datetime.timedelta(minutes=time_last_timetype.minute + 1)
                        else:
                            # 生成一个 0 到 3 分钟之间的随机数
                            random_minutes = random.randint(0, 3)
                            # 将随机数加到时间戳上
                            new_time = datetime.datetime.combine(datetime.date.today(), origin_time) + datetime.timedelta(minutes=random_minutes)
                            # 将新的 datetime 对象转换回 time 对象
                            new_time = new_time.time()
                            # 将新的时间转换为字符串

                        time = new_time.strftime("%H:%M:%S")
                    else:
                        table, sql, characteristic, value, sql_id = get_sql_markov(user_id, sql_id, exe_sql_num)

                        origin_time = datetime.datetime.strptime(time, "%H:%M:%S").time()
                        # 生成一个 10 到 40 秒之间的随机数
                        random_seconds = gaussian_tpcc(33, 11)
                        # 将随机数加到时间戳上
                        new_datetime = datetime.datetime.combine(datetime.date.today(), origin_time) + datetime.timedelta(seconds=random_seconds)
                        # 将新的 datetime 对象转换回 time 对象
                        new_time = new_datetime.time()
                        # 将新的时间转换为字符串
                        time = new_time.strftime("%H:%M:%S")

                        if exe_sql_num == sqls_period - 1:
                            time_last = time


                    print(f"user_id: {user_id}, week is: {week_cnt}, "
                          f"Today is: {day}, sqls in today: {int(sql_num)}, "
                          f"time is :{time}, sqls in this time is :{sqls_period}, "
                          f"sql now: {exe_sql_num + 1}, sql is :{sql}, sql_id is: {sql_id}, value is:{value}")

                    cursor.execute(sql, value)
                    if characteristic[0] == 'select':  #计算selectivity
                        result = cursor.fetchall()
                        selectivity = len(result) / rows_table[table]
                        sql_character[columns.index(table + "-selectivity")] = selectivity

                    sql_character[columns.index("sql_id")] = sql_id
                    sql_character[columns.index(characteristic[0])] += 1
                    sql_character[columns.index(table)] += 1
                    print(f"sql type:{characteristic[0]}, num: {sql_character[columns.index(characteristic[0])]}")
                    for col in characteristic[1]:
                        sql_character[columns.index(col)] += 1
                    for col in characteristic[2]:
                        sql_character[columns.index(col)] += 1

                    time_index1.append(date_time)
                    time_index1.append(time)
                    time_index2.append(days_to_add)
                    time_index2.append(time_period + 1)

                    sql_character[columns.index("time_index1")] = time_index1
                    sql_character[columns.index("time_index2")] = time_index2

                    data.append(sql_character)

    df = pd.DataFrame(data, columns=columns)
    csv = str(user_id) + 'sql_statements.csv'
    df.to_csv(csv, index=False)

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    pool = Pool(client)
    for i in range(client):
        # pool.apply_async(execute_sql,(i,))
        pool.apply(execute_sql,(i,))
    pool.close()
    pool.join()

    # test = [[0.4, 0.6], [0.2, 0.8]]
    # print(np.array(test) ** 5)
    #
    # it = np.dot(np.array(p_markov[Users[2]][30]), np.array(p_markov[Users[2]]))
    # # print(numpy.sum(np.array(p_markov[Users[0]])))
    # #
    # print(it, "\n", sum(it))
    #
    # for i in range(100):
    #     random_number = random.random() * sum(it)
    #     print(random_number)
    #
    #     cumulative_sum = 0
    #     counter = 0
    #     # 遍历列表，累加元素值，直到累加和超过随机数
    #     for num in it:
    #         cumulative_sum += num
    #         counter += 1
    #         if cumulative_sum >= random_number:
    #             break
    #     print(counter)

    # for i in range(3):
    #     print(len(p_markov[Users[i]]))
    #     print(p_markov[Users[i]][0], "\n", np.dot(np.array(p_markov[Users[i]][0]), np.array(p_markov[Users[i]])))
    # for i in range(len(p_FinancialMan)):
    #     print(i + 1, len(p_FinancialMan[i]))
    #
    # print(p_FinancialMan[0], "\n", np.dot(np.array(p_FinancialMan[0]), np.array(p_FinancialMan)))


