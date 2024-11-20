import random
import string
import random
from datetime import datetime, timedelta
import constants


def set_value(type, column):
    if type == "insert":
        if column == "C_ID":
            return random.randint(3001, 30000)
        elif column == "C_D_ID":
            return random.randint(10, 10000)
        elif column == "C_W_ID":
            return random.randint(10, 10000)
        elif column == "D_ID":
            return random.randint(10, 10000)
        elif column == "D_W_ID":
            return random.randint(10, 10000)
        elif column == "I_ID":
            return random.randint(100000, 100000000)
        elif column == "O_W_ID":
            return random.randint(10, 100000)
        elif column == "O_D_ID":
            return random.randint(10, 10000)
        elif column == "O_ID":
            return random.randint(3000, 3000000)
        elif column == "OL_O_ID":
            return random.randint(3000, 3000000)
        elif column == "OL_D_ID":
            return random.randint(10, 10000)
        elif column == "OL_W_ID":
            return random.randint(10, 10000)
        elif column == "OL_NUMBER":
            return random.randint(15, 10000)
        elif column == "S_W_ID":
            return random.randint(10, 10000)
        elif column == "S_I_ID":
            return random.randint(100000, 100000000)
        elif column == "W_ID":
            return random.randint(3000, 3000000)
    else:
        if column == "C_ID":
            return random.randint(1, 3000)
        elif column == "C_D_ID":
            return random.randint(1, 10)
        elif column == "C_W_ID":
            return random.randint(1, 10)
        elif column == "C_CREDIT_LIM":
            return random.randint(1, 3000)
        elif column == "C_DISCOUNT":
            return random.randint(1, 5000)
        elif column == "C_BALANCE":
            return random.randint(1, 10)
        elif column == "C_YTD_PAYMENT":
            return random.randint(1, 5)
        elif column == "C_PAYMENT_CNT":
            return random.randint(1, 5)
        elif column == "C_DELIVERY_CNT":
            return random.randint(1, 5)
        elif column == "D_ID":
            return random.randint(1, 10)
        elif column == "D_W_ID":
            return random.randint(1, 10)
        elif column == "D_TAX":
            return random.randint(1, 100)
        elif column == "D_YTD":
            return random.randint(1, 5)
        elif column == "D_NEXT_O_ID":
            return random.randint(1, 3)
        elif column == "H_C_ID":
            return random.randint(1, 3000)
        elif column == "H_C_D_ID":
            return random.randint(1, 10)
        elif column == "H_C_W_ID":
            return random.randint(1, 10)
        elif column == "H_D_ID":
            return random.randint(1, 10)
        elif column == "H_W_ID":
            return random.randint(1, 10)
        elif column == "H_DATE":
            start_date = datetime(2024, 11, 1)
            end_date = datetime(2024, 11, 10)
            delta = end_date - start_date
            random_days = random.randint(0, delta.days)
            random_date = start_date + timedelta(days=random_days)
            return random_date
        elif column == "H_AMOUNT":
            return random.randint(1, 5)
        elif column == "I_ID":
            return random.randint(1, 100000)
        elif column == "I_IM_ID":
            return random.randint(1, 10000)
        elif column == "I_PRICE":
            return random.randint(1, 9800)
        elif column == "O_W_ID":
            return random.randint(1, 10)
        elif column == "O_D_ID":
            return random.randint(1, 10)
        elif column == "O_ID":
            return random.randint(1, 3000)
        elif column == "O_C_ID":
            return random.randint(1, 3000)
        elif column == "O_OL_CNT":
            return random.randint(1, 11)
        elif column == "O_ALL_LOCAL":
            return random.randint(1, 3)
        elif column == "O_CARRIER_ID":
            return random.randint(1, 11)
        elif column == "OL_O_ID":
            return random.randint(1, 3000)
        elif column == "OL_D_ID":
            return random.randint(1, 10)
        elif column == "OL_W_ID":
            return random.randint(1, 10)
        elif column == "OL_NUMBER":
            return random.randint(1, 15)
        elif column == "OL_I_ID":
            return random.randint(1, 100000)
        elif column == "OL_SUPPLY_W_ID":
            return random.randint(1, 10)
        elif column == "OL_AMOUNT":
            return random.randint(1, 99000)
        elif column == "S_W_ID":
            return random.randint(1, 10)
        elif column == "S_I_ID":
            return random.randint(1, 100000)
        elif column == "S_QUANTITY":
            return random.randint(1, 90)
        elif column == "S_YTD":
            return random.randint(1, 3)
        elif column == "S_ORDER_CNT":
            return random.randint(1, 3)
        elif column == "S_REMOTE_CNT":
            return random.randint(1, 3)
        elif column == "W_ID":
            return random.randint(1, 3000)
        elif column == "W_NAME":
            return constants.columns_tables["WAREHOUSE"]["W_NAME"][random.randint(1, 10) - 1]
        elif column == "W_STREET_1":
            return constants.columns_tables["WAREHOUSE"]["W_STREET_1"][random.randint(1, 10) - 1]
        elif column == "W_STREET_2":
            return constants.columns_tables["WAREHOUSE"]["W_STREET_2"][random.randint(1, 10) - 1]
        elif column == "W_CITY":
            return constants.columns_tables["WAREHOUSE"]["W_CITY"][random.randint(1, 10) - 1]
        elif column == "W_STATE":
            return constants.columns_tables["WAREHOUSE"]["W_STATE"][random.randint(1, 10) - 1]
        elif column == "W_ZIP":
            return constants.columns_tables["WAREHOUSE"]["W_ZIP"][random.randint(1, 10) - 1]
        elif column == "W_TAX":
            return constants.columns_tables["WAREHOUSE"]["W_TAX"][random.randint(1, 10) - 1]
        elif column == "W_YTD":
            return 300000
        else:
            return ''.join(random.choice(string.ascii_letters) for _ in range(5))
    