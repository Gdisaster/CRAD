Users = ["SalesMan", "WarehousesMan", "FinancialMan"]

Tables = {
    "SalesMan": ["CUSTOMER", "ORDERS", "ORDER_LINE", "HISTORY", "ITEM", "STOCK", "WAREHOUSE", "DISTRICT"],
    "WarehousesMan": ["STOCK", "WAREHOUSE", "ORDERS", "ORDER_LINE", "DISTRICT", "ITEM", "CUSTOMER"],
    "FinancialMan": ["HISTORY", "ORDERS", "ORDER_LINE", "CUSTOMER", "STOCK", "ITEM", "WAREHOUSE", "DISTRICT"]
}

SQLS_user_table_id = {
    "SalesMan": {
        "CUSTOMER": {
            1: ["""SELECT C_FIRST, C_LAST,C_PHONE ,C_DATA
                    FROM CUSTOMER 
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;
                    """],
            2: ["""SELECT C_STREET_1, C_STREET_2, C_CITY, C_STATE, C_ZIP
                    FROM CUSTOMER
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;
                    """],
            3: ["""SELECT C_DISCOUNT
                    FROM CUSTOMER
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;
                    """],
            4: ["""SELECT C_BALANCE
                    FROM CUSTOMER
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;
                    """],
            5: ["""SELECT C_CREDIT,C_CREDIT_LIM
                    FROM CUSTOMER 
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;
                    """],
            6: ["""SELECT C_SINCE
                    FROM CUSTOMER
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;
                    """],
            7: ["""INSERT INTO CUSTOMER (
                    C_ID, C_D_ID, C_W_ID, C_FIRST, C_MIDDLE, C_LAST, C_STREET_1, C_STREET_2, 
                    C_CITY, C_STATE, C_ZIP, C_PHONE, C_SINCE, C_CREDIT, C_CREDIT_LIM, 
                    C_DISCOUNT, C_BALANCE, C_YTD_PAYMENT, C_PAYMENT_CNT, C_DELIVERY_CNT, C_DATA
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """],
            8: ["""UPDATE CUSTOMER 
                    SET C_DISCOUNT = %s 
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;
                    """],
            9: ["""UPDATE CUSTOMER 
                    SET C_CREDIT = %s ,C_CREDIT_LIM = %s 
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;
                    """],
            10: ["""UPDATE CUSTOMER
                    SET C_STREET_1 = %s, C_STREET_2 = %s, C_CITY = %s, C_STATE = %s, C_ZIP = %s
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;
                    """],
            11: ["""DELETE FROM CUSTOMER 
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;
                    """],
        },
        "ORDERS": {
            1: ["""INSERT INTO ORDERS (
                    O_W_ID, O_D_ID, O_ID, O_C_ID, O_ENTRY_D, O_OL_CNT, O_ALL_LOCAL, O_CARRIER_ID)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s);
                    """],
            2: ["""SELECT O_C_ID 
                    FROM ORDERS
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;
                    """],
            3: ["""SELECT O_W_ID, O_D_ID
                    FROM ORDERS
                    WHERE O_ID = %s;
                    """],
            4: ["""SELECT O_CARRIER_ID
                    FROM ORDERS
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;
                    """],
            5: ["""SELECT O_OL_CNT 
                    FROM ORDERS
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;
                    """],
            6: ["""SELECT O_ID,O_ENTRY_D 
                    FROM ORDERS
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;
                    """],
            7: ["""UPDATE ORDERS
                    SET O_CARRIER_ID= %s
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;
                    """],
            8: ["""DELETE FROM ORDERS
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;
                    """],
        },
        "ORDER_LINE": {
            1: ["""SELECT OL_O_ID, OL_I_ID, OL_QUANTITY
                    FROM ORDER_LINE
                    WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s;
                    """],
            2: ["""SELECT OL_O_ID, OL_I_ID, OL_AMOUNT
                    FROM ORDER_LINE
                    WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s;
                    """],
            3: ["""UPDATE ORDER_LINE
                    SET OL_QUANTITY = %s, OL_AMOUNT = %s
                    WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s;
                    """],
            4: ["""INSERT INTO ORDER_LINE (
                    OL_O_ID, OL_D_ID, OL_W_ID, OL_NUMBER, OL_I_ID, OL_SUPPLY_W_ID, OL_QUANTITY,
                    OL_AMOUNT, OL_DIST_INFO, OL_DELIVERY_D)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, TIMESTAMP);
                    """],
            5: ["""UPDATE ORDER_LINE
                    SET OL_AMOUNT = %s
                    WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s;
                    """],
            6: ["""DELETE FROM ORDER_LINE
                    WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s;
                    """],
        },
        "HISTORY": {
            1: ["""SELECT SUM(H_AMOUNT) AS TOTAL_AMOUNT
                    FROM HISTORY
                    WHERE H_C_ID = %s AND H_DATE BETWEEN %s AND %s;
                    """],
            2: ["""SELECT H_DATE,H_AMOUNT
                    FROM HISTORY
                    WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s
                    AND H_DATE BETWEEN %s AND %s;
                    """],
            3: ["""SELECT H_DATE,H_AMOUNT,H_DATA 
                    FROM HISTORY
                    WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s
                    AND H_DATE BETWEEN %s AND %s
                    """],
            4: ["""INSERT INTO HISTORY (
                    H_C_ID, H_C_D_ID, H_C_W_ID, H_D_ID, H_W_ID, H_DATE, H_AMOUNT, H_DATA)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s)
                    """],
            5: ["""UPDATE HISTORY
                    SET H_AMOUNT = %s, H_DATA = %s
                    WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s
                    """],
            6: ["""DELETE FROM HISTORY
                    WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s
                    """],
        },
        "ITEM": {
            1: ["""SELECT I_ID, I_IM_ID
                    FROM ITEM
                    WHERE I_NAME = %s
                    """],
            2: ["""SELECT I_PRICE, I_DATA
                    FROM ITEM
                    WHERE I_NAME = %s
                    """],
            3: ["""SELECT I_NAME
                    FROM ITEM
                    WHERE I_ID = %s;
                    """],
        },
        "STOCK": {
            1: ["""SELECT S_QUANTITY 
                    FROM STOCK 
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
            2: ["""SELECT S_ORDER_CNT 
                    FROM STOCK 
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
            3: ["""SELECT S_YTD 
                    FROM STOCK 
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
        },
        "WAREHOUSE": {
            1: ["""SELECT W_YTD 
                    FROM WAREHOUSE 
                    WHERE W_ID = %s
                    """],
            2: ["""SELECT W_TAX 
                    FROM WAREHOUSE 
                    WHERE W_ID = %s
                    """],
        },
        "DISTRICT": {
            1: ["""SELECT D_YTD
                    FROM DISTRICT
                    WHERE D_ID = %s AND D_W_ID = %s
                    """],
            2: ["""SELECT D_TAX
                    FROM DISTRICT
                    WHERE D_ID = %s AND D_W_ID = %s
                    """],
        },
    },
    "WarehousesMan": {
        "STOCK": {
            1: ["""SELECT S_QUANTITY
                    FROM STOCK 
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
            2: ["""SELECT S_ORDER_CNT
                    FROM STOCK 
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
            3: ["""SELECT S_QUANTITY,S_DATA
                    FROM STOCK 
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
            4: ["""SELECT S_I_ID, S_QUANTITY
                    FROM STOCK
                    WHERE S_W_ID = %s AND S_QUANTITY < 10
                    """],
            5: ["""SELECT S_DIST_01, S_DIST_02, S_DIST_03, S_DIST_04, S_DIST_05, S_DIST_06, S_DIST_07, S_DIST_08, S_DIST_09, S_DIST_10
                    FROM STOCK
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
            6: ["""UPDATE STOCK
                    SET S_QUANTITY = %s, S_YTD = %s, S_ORDER_CNT = %s, 
                    S_REMOTE_CNT = %s, S_DATA = %s
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
            7: ["""UPDATE STOCK
                    SET
                    S_DIST_01 = %s, S_DIST_02 = %s, S_DIST_03 = %s, S_DIST_04 = %s,
                    S_DIST_05 = %s, S_DIST_06 = %s, S_DIST_07 = %s, S_DIST_08 = %s,
                    S_DIST_09 = %s, S_DIST_10 = %s
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
            8: ["""INSERT INTO STOCK (
                    S_W_ID, S_I_ID, S_QUANTITY, S_YTD, S_ORDER_CNT, S_REMOTE_CNT, S_DATA, 
                    S_DIST_01, S_DIST_02, S_DIST_03, S_DIST_04, S_DIST_05, S_DIST_06, S_DIST_07, S_DIST_08, S_DIST_09, S_DIST_10)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """],
            9: ["""DELETE FROM STOCK
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
        },
        "WAREHOUSE": {
            1: ["""SELECT W_STREET_1, W_STREET_2, W_CITY, W_STATE, W_ZIP
                    FROM WAREHOUSE
                    WHERE W_ID = %s
                    """],
            2: ["""SELECT W_ID, W_NAME
                    FROM WAREHOUSE
                    WHERE W_CITY = %s
                    """],
            3: ["""SELECT W_ID, W_NAME
                    FROM WAREHOUSE
                    WHERE W_STATE = %s
                    """],
            4: ["""INSERT INTO WAREHOUSE (
                    W_ID, W_NAME, W_STREET_1, W_STREET_2, W_CITY, W_STATE, W_ZIP, W_TAX, W_YTD)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """],
            5: ["""UPDATE WAREHOUSE
                    SET W_NAME = %s, W_STREET_1 = %s, W_STREET_2 = %s, W_CITY = %s, W_STATE = %s, W_ZIP = %s
                    WHERE W_ID = %s
                    """],
            6: ["""DELETE FROM WAREHOUSE
                    WHERE W_ID = %s AND W_NAME = %s
                    """],
        },
        "ORDERS": {
            1: ["""SELECT O_W_ID, O_D_ID
                    FROM ORDERS
                    WHERE O_ID = %s
                    """],
            2: ["""SELECT O_CARRIER_ID
                    FROM ORDERS
                    WHERE O_ID = %s
                    """],
            3: ["""SELECT O_OL_CNT
                    FROM ORDERS
                    WHERE O_ID = %s AND O_W_ID = %s AND O_D_ID = %s
                    """],
            4: ["""UPDATE ORDERS
                    SET O_CARRIER_ID= %s
                    WHERE O_ID = %s
                    """],
        },
        "ORDER_LINE": {
            1: ["""SELECT OL_I_ID, OL_SUPPLY_W_ID, OL_DELIVERY_D, OL_QUANTITY, OL_DIST_INFO
                    FROM ORDER_LINE
                    WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s
                    """],
            2: ["""SELECT OL_D_ID,OL_W_ID
                    FROM ORDER_LINE
                    WHERE OL_O_ID = %s
                    """],
            3: ["""UPDATE ORDER_LINE
                    SET OL_DELIVERY_D = %s, OL_DIST_INFO = %s
                    WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s
                    """],
        },
        "DISTRICT": {
            1: ["""SELECT D_STREET_1, D_STREET_2, D_CITY, D_STATE, D_ZIP
                    FROM DISTRICT
                    WHERE D_ID = %s
                    """],
            2: ["""SELECT D_W_ID
                    FROM DISTRICT
                    WHERE D_ID = %s
                    """],
            3: ["""SELECT D_ID, D_NAME
                    FROM DISTRICT
                    WHERE D_CITY = %s
                    """],
            4: ["""SELECT D_ID, D_NAME
                    FROM DISTRICT
                    WHERE D_STATE = %s
                    """],
            5: ["""INSERT INTO DISTRICT (
                    D_ID, D_W_ID, D_NAME, D_STREET_1, D_STREET_2, D_CITY, D_STATE, D_ZIP, D_TAX, D_YTD, D_NEXT_O_ID)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """],
            6: ["""UPDATE DISTRICT
                    SET D_NAME = %s, D_STREET_1 = %s, D_STREET_2 = %s, D_CITY = %s, D_STATE = %s, D_ZIP = %s
                    WHERE D_ID = %s AND D_W_ID = %s
                    """],
            7: ["""DELETE FROM DISTRICT
                    WHERE D_ID = %s AND D_W_ID = %s AND D_NAME = %s
                    """],
        },
        "ITEM": {
            1: ["""SELECT * 
                    FROM ITEM
                    WHERE I_ID = %s
                    """],
            2: ["""UPDATE ITEM
                    SET I_NAME = %s, I_PRICE = %s, I_DATA = %s
                    WHERE I_ID = %s
                    """],
            3: ["""INSERT INTO ITEM (
                    I_ID, I_IM_ID, I_NAME, I_PRICE, I_DATA)
                    VALUES (%s, %s, %s, %s, %s)
                    """],
            4: ["""DELETE FROM ITEM
                    WHERE I_ID = %s
                    """],
        },
        "CUSTOMER": {
            1: ["""SELECT C_STREET_1, C_STREET_2, C_CITY, C_STATE, C_ZIP, C_PHONE
                    FROM CUSTOMER
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s
                    """],
        },
    },

    "FinancialMan": {
        "HISTORY": {
            1: ["""SELECT SUM(H_AMOUNT) AS TOTAL_AMOUNT
                    FROM HISTORY
                    WHERE H_C_ID = %s AND H_DATE BETWEEN %s AND %s
                    """],
            2: ["""SELECT H_DATE,H_AMOUNT
                    FROM HISTORY
                    WHERE H_C_ID = %s AND H_DATE BETWEEN %s AND %s
                    """],
            3: ["""SELECT SUM(H_AMOUNT) AS TOTAL_AMOUNT
                    FROM HISTORY 
                    WHERE H_C_D_ID = %s AND H_DATE BETWEEN %s AND %s
                    """],
            4: ["""SELECT H_DATE,H_AMOUNT
                    FROM HISTORY 
                    WHERE H_C_D_ID = %s AND H_DATE BETWEEN %s AND %s
                    """],
            5: ["""SELECT SUM(H_AMOUNT) AS TOTAL_AMOUNT
                    FROM HISTORY 
                    WHERE H_W_ID = %s AND H_DATE BETWEEN %s AND %s
                    """],
            6: ["""SELECT H_DATE,H_AMOUNT
                    FROM HISTORY 
                    WHERE H_W_ID = %s AND H_DATE BETWEEN %s AND %s
                    """],
            7: ["""UPDATE HISTORY
                    SET H_AMOUNT = %s, H_DATA = %s
                    WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s
                    """],
            8: ["""DELETE FROM HISTORY
                    WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s
                    """],
        },
        "ORDERS": {
            1: ["""SELECT O_C_ID 
                    FROM ORDERS
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s
                    """],
            2: ["""SELECT O_CARRIER_ID
                    FROM ORDERS
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s
                    """],
            3: ["""SELECT O_ID,O_ENTRY_D 
                    FROM ORDERS
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s
                    """],
            4: ["""SELECT O_OL_CNT 
                    FROM ORDERS
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s
                    """],
            5: ["""SELECT O_W_ID, O_D_ID
                    FROM ORDERS
                    WHERE O_ID = %s
                    """],
            6: ["""INSERT INTO ORDERS (
                    O_W_ID, O_D_ID, O_ID, O_C_ID, O_ENTRY_D, O_OL_CNT, O_ALL_LOCAL, O_CARRIER_ID_ID)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s)
                    """],
            7: ["""UPDATE ORDERS
                    SET O_CARRIER_ID= %s
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s
                    """],
            8: ["""DELETE FROM ORDERS
                    WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s
                    """],
        },
        "ORDER_LINE": {
            1: ["""SELECT OL_AMOUNT
                    FROM ORDER_LINE
                    WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s
                    """],
            2: ["""SELECT SUM(OL_QUANTITY) AS TotalQuantity
                    FROM ORDER_LINE
                    WHERE OL_O_ID = %s AND OL_DELIVERY_D BETWEEN %s AND %s
                    """],
            3: ["""SELECT SUM(OL_AMOUNT) AS TotalSales
                    FROM ORDER_LINE
                    WHERE OL_DELIVERY_D BETWEEN %s AND %s
                    """],
        },
        "CUSTOMER": {
            1: ["""SELECT C_FIRST, C_LAST, C_PHONE 
                    FROM CUSTOMER 
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s
                    """],
            2: ["""SELECT C_CREDIT,C_CREDIT_LIM
                    FROM CUSTOMER 
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s
                    """],
            3: ["""SELECT C_CREDIT_LIM, C_DISCOUNT
                    FROM CUSTOMER
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s
                    """],
            4: ["""SELECT C_SINCE, C_YTD_PAYMENT
                    FROM CUSTOMER
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s
                    """],
            5: ["""SELECT C_YTD_PAYMENT,C_PAYMENT_CNT, C_BALANCE
                    FROM CUSTOMER
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s
                    """],
            6: ["""UPDATE CUSTOMER 
                    SET C_CREDIT = %s, C_CREDIT_LIM = %s 
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s
                    """],
            7: ["""UPDATE CUSTOMER 
                    SET C_DISCOUNT = %s 
                    WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s
                    """],
        },
        "STOCK": {
            1: ["""SELECT S_QUANTITY, S_ORDER_CNT
                    FROM STOCK
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
            2: ["""SELECT S_I_ID, S_QUANTITY, S_REMOTE_CNT
                    FROM STOCK
                    WHERE S_W_ID = %s AND S_REMOTE_CNT > 0
                    """],
            3: ["""SELECT S_QUANTITY, S_ORDER_CNT, S_REMOTE_CNT
                    FROM STOCK
                    WHERE S_I_ID = %s AND S_W_ID = %s
                    """],
            4: ["""SELECT S_YTD
                    FROM STOCK
                    WHERE S_W_ID = %s AND S_I_ID = %s
                    """],
        },
        "ITEM": {
            1: ["""SELECT I_PRICE
                    FROM ITEM
                    WHERE I_ID = %s
                    """],
        },
        "WAREHOUSE": {
            1: ["""SELECT W_NAME, W_YTD
                    FROM WAREHOUSE
                    WHERE W_ID = %s
                    """],
            2: ["""SELECT W_NAME, W_TAX
                    FROM WAREHOUSE
                    WHERE W_ID = %s
                    """],
            3: ["""SELECT W_STREET_1, W_STREET_2, W_CITY, W_STATE, W_ZIP
                    FROM WAREHOUSE
                    WHERE W_ID = %s
                    """],
            4: ["""UPDATE WAREHOUSE
                    SET W_TAX = %s, W_YTD = %s
                    WHERE W_ID = %s
                    """],
        },
        "DISTRICT": {
            1: ["""SELECT D_NAME, D_YTD
                    FROM DISTRICT
                    WHERE D_ID = %s
                    """],
            2: ["""SELECT D_NAME, D_TAX
                    FROM DISTRICT
                    WHERE D_ID = %s
                    """],
            3: ["""SELECT D_STREET_1, D_STREET_2, D_CITY, D_STATE, D_ZIP
                    FROM DISTRICT
                    WHERE D_W_ID = %s AND D_ID = %s
                    """],
            4: ["""UPDATE DISTRICT
                    SET D_TAX = %s, D_YTD = %s
                    WHERE D_ID = %s AND D_W_ID = %s
                    """],
        },
    },
}

SQLS_user_id = {
    "SalesMan": {
        1: """SELECT C_FIRST, C_LAST,C_PHONE ,C_DATA FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;""",
        2: """SELECT C_STREET_1, C_STREET_2, C_CITY, C_STATE, C_ZIP FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;""",
        3: """SELECT C_DISCOUNT FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;""",
        4: """SELECT C_BALANCE FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;""",
        5: """SELECT C_CREDIT,C_CREDIT_LIM FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;""",
        6: """SELECT C_SINCE FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;""",
        7: """INSERT INTO CUSTOMER (C_ID, C_D_ID, C_W_ID, C_FIRST, C_MIDDLE, C_LAST, C_STREET_1, C_STREET_2, C_CITY, C_STATE, C_ZIP, C_PHONE, C_SINCE, C_CREDIT, C_CREDIT_LIM, C_DISCOUNT, C_BALANCE, C_YTD_PAYMENT, C_PAYMENT_CNT, C_DELIVERY_CNT, C_DATA) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""",
        8: """UPDATE CUSTOMER SET C_DISCOUNT = %s WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;""",
        9: """UPDATE CUSTOMER SET C_CREDIT = %s ,C_CREDIT_LIM = %s WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;""",
        10: """UPDATE CUSTOMER SET C_STREET_1 = %s, C_STREET_2 = %s, C_CITY = %s, C_STATE = %s, C_ZIP = %s WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;""",
        11: """DELETE FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s;""",
        12: """INSERT INTO ORDERS (O_W_ID, O_D_ID, O_ID, O_C_ID, O_ENTRY_D, O_OL_CNT, O_ALL_LOCAL, O_CARRIER_ID) VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s);""",
        13: """SELECT O_C_ID FROM ORDERS WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;""",
        14: """SELECT O_W_ID, O_D_ID FROM ORDERS WHERE O_ID = %s;""",
        15: """SELECT O_CARRIER_ID FROM ORDERS WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;""",
        16: """SELECT O_OL_CNT FROM ORDERS WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;""",
        17: """SELECT O_ID,O_ENTRY_D FROM ORDERS WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;""",
        18: """UPDATE ORDERS SET O_CARRIER_ID= %s WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;""",
        19: """DELETE FROM ORDERS WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s;""",
        20: """SELECT OL_O_ID, OL_I_ID, OL_QUANTITY FROM ORDER_LINE WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s;""",
        21: """SELECT OL_O_ID, OL_I_ID, OL_AMOUNT FROM ORDER_LINE WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s;""",
        22: """UPDATE ORDER_LINE SET OL_QUANTITY = %s, OL_AMOUNT = %s WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s;""",
        23: """INSERT INTO ORDER_LINE (OL_O_ID, OL_D_ID, OL_W_ID, OL_NUMBER, OL_I_ID, OL_SUPPLY_W_ID, OL_QUANTITY, OL_AMOUNT, OL_DIST_INFO, OL_DELIVERY_D) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, TIMESTAMP);""",
        24: """UPDATE ORDER_LINE SET OL_AMOUNT = %s WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s;""",
        25: """DELETE FROM ORDER_LINE WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s;""",
        26: """SELECT SUM(H_AMOUNT) AS TOTAL_AMOUNT FROM HISTORY WHERE H_C_ID = %s AND H_DATE BETWEEN %s AND %s;""",
        27: """SELECT H_DATE,H_AMOUNT FROM HISTORY WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s AND H_DATE BETWEEN %s AND %s;""",
        28: """SELECT H_DATE,H_AMOUNT,H_DATA FROM HISTORY WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s AND H_DATE BETWEEN %s AND %s;""",
        29: """INSERT INTO HISTORY (H_C_ID, H_C_D_ID, H_C_W_ID, H_D_ID, H_W_ID, H_DATE, H_AMOUNT, H_DATA) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s);""",
        30: """UPDATE HISTORY SET H_AMOUNT = %s, H_DATA = %s WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s;""",
        31: """DELETE FROM HISTORY WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s;""",
        32: """SELECT I_ID, I_IM_ID FROM ITEM WHERE I_NAME = %s;""",
        33: """SELECT I_PRICE, I_DATA FROM ITEM WHERE I_NAME = %s;""",
        34: """SELECT I_NAME FROM ITEM WHERE I_ID = %s;""",
        35: """SELECT S_QUANTITY FROM STOCK WHERE S_W_ID = %s AND S_I_ID = %s;""",
        36: """SELECT S_ORDER_CNT FROM STOCK WHERE S_W_ID = %s AND S_I_ID = %s;""",
        37: """SELECT S_YTD FROM STOCK WHERE S_W_ID = %s AND S_I_ID = %s;""",
        38: """SELECT W_YTD FROM WAREHOUSE WHERE W_ID = %s;""",
        39: """SELECT W_TAX FROM WAREHOUSE WHERE W_ID = %s;""",
        40: """SELECT D_YTD FROM DISTRICT WHERE D_ID = %s AND D_W_ID = %s;""",
        41: """SELECT D_TAX FROM DISTRICT WHERE D_ID = %s AND D_W_ID = %s;"""
    },
    "WarehousesMan": {
        1: """SELECT S_QUANTITY FROM STOCK WHERE S_W_ID = %s AND S_I_ID = %s""",
        2: """SELECT S_ORDER_CNT FROM STOCK WHERE S_W_ID = %s AND S_I_ID = %s""",
        3: """SELECT S_QUANTITY,S_DATA FROM STOCK WHERE S_W_ID = %s AND S_I_ID = %s""",
        4: """SELECT S_I_ID, S_QUANTITY FROM STOCK WHERE S_W_ID = %s AND S_QUANTITY < 10""",
        5: """SELECT S_DIST_01, S_DIST_02, S_DIST_03, S_DIST_04, S_DIST_05, S_DIST_06, S_DIST_07, S_DIST_08, S_DIST_09, S_DIST_10 FROM STOCK WHERE S_W_ID = %s AND S_I_ID = %s""",
        6: """UPDATE STOCK SET S_QUANTITY = %s, S_YTD = %s, S_ORDER_CNT = %s, S_REMOTE_CNT = %s, S_DATA = %s WHERE S_W_ID = %s AND S_I_ID = %s""",
        7: """UPDATE STOCK SET S_DIST_01 = %s, S_DIST_02 = %s, S_DIST_03 = %s, S_DIST_04 = %s, S_DIST_05 = %s, S_DIST_06 = %s, S_DIST_07 = %s, S_DIST_08 = %s, S_DIST_09 = %s, S_DIST_10 = %s WHERE S_W_ID = %s AND S_I_ID = %s""",
        8: """INSERT INTO STOCK (S_W_ID, S_I_ID, S_QUANTITY, S_YTD, S_ORDER_CNT, S_REMOTE_CNT, S_DATA, S_DIST_01, S_DIST_02, S_DIST_03, S_DIST_04, S_DIST_05, S_DIST_06, S_DIST_07, S_DIST_08, S_DIST_09, S_DIST_10) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        9: """DELETE FROM STOCK WHERE S_W_ID = %s AND S_I_ID = %s""",
        10: """SELECT W_STREET_1, W_STREET_2, W_CITY, W_STATE, W_ZIP FROM WAREHOUSE WHERE W_ID = %s""",
        11: """SELECT W_ID, W_NAME FROM WAREHOUSE WHERE W_CITY = %s""",
        12: """SELECT W_ID, W_NAME FROM WAREHOUSE WHERE W_STATE = %s""",
        13: """INSERT INTO WAREHOUSE (W_ID, W_NAME, W_STREET_1, W_STREET_2, W_CITY, W_STATE, W_ZIP, W_TAX, W_YTD) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        14: """UPDATE WAREHOUSE SET W_NAME = %s, W_STREET_1 = %s, W_STREET_2 = %s, W_CITY = %s, W_STATE = %s, W_ZIP = %s WHERE W_ID = %s""",
        15: """DELETE FROM WAREHOUSE WHERE W_ID = %s AND W_NAME = %s""",
        16: """SELECT O_W_ID, O_D_ID FROM ORDERS WHERE O_ID = %s""",
        17: """SELECT O_CARRIER_ID FROM ORDERS WHERE O_ID = %s""",
        18: """SELECT O_OL_CNT FROM ORDERS WHERE O_ID = %s AND O_W_ID = %s AND O_D_ID = %s""",
        19: """UPDATE ORDERS SET O_CARRIER_ID= %s WHERE O_ID = %s""",
        20: """SELECT OL_I_ID, OL_SUPPLY_W_ID, OL_DELIVERY_D, OL_QUANTITY, OL_DIST_INFO FROM ORDER_LINE WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s""",
        21: """SELECT OL_D_ID,OL_W_ID FROM ORDER_LINE WHERE OL_O_ID = %s""",
        22: """UPDATE ORDER_LINE SET OL_DELIVERY_D = %s, OL_DIST_INFO = %s WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s AND OL_NUMBER = %s""",
        23: """SELECT D_STREET_1, D_STREET_2, D_CITY, D_STATE, D_ZIP FROM DISTRICT WHERE D_ID = %s""",
        24: """SELECT D_W_ID FROM DISTRICT WHERE D_ID = %s""",
        25: """SELECT D_ID, D_NAME FROM DISTRICT WHERE D_CITY = %s""",
        26: """SELECT D_ID, D_NAME FROM DISTRICT WHERE D_STATE = %s""",
        27: """INSERT INTO DISTRICT (D_ID, D_W_ID, D_NAME, D_STREET_1, D_STREET_2, D_CITY, D_STATE, D_ZIP, D_TAX, D_YTD, D_NEXT_O_ID) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        28: """UPDATE DISTRICT SET D_NAME = %s, D_STREET_1 = %s, D_STREET_2 = %s, D_CITY = %s, D_STATE = %s, D_ZIP = %s WHERE D_ID = %s AND D_W_ID = %s""",
        29: """DELETE FROM DISTRICT WHERE D_ID = %s AND D_W_ID = %s AND D_NAME = %s""",
        30: """SELECT * FROM ITEM WHERE I_ID = %s""",
        31: """UPDATE ITEM SET I_NAME = %s, I_PRICE = %s, I_DATA = %s WHERE I_ID = %s""",
        32: """INSERT INTO ITEM (I_ID, I_IM_ID, I_NAME, I_PRICE, I_DATA) VALUES (%s, %s, %s, %s, %s)""",
        33: """DELETE FROM ITEM WHERE I_ID = %s""",
        34: """SELECT C_STREET_1, C_STREET_2, C_CITY, C_STATE, C_ZIP, C_PHONE FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s"""
    },

    "FinancialMan": {
        1: """SELECT SUM(H_AMOUNT) AS TOTAL_AMOUNT FROM HISTORY WHERE H_C_ID = %s AND H_DATE BETWEEN %s AND %s""",
        2: """SELECT H_DATE,H_AMOUNT FROM HISTORY WHERE H_C_ID = %s AND H_DATE BETWEEN %s AND %s""",
        3: """SELECT SUM(H_AMOUNT) AS TOTAL_AMOUNT FROM HISTORY WHERE H_C_D_ID = %s AND H_DATE BETWEEN %s AND %s""",
        4: """SELECT H_DATE,H_AMOUNT FROM HISTORY WHERE H_C_D_ID = %s AND H_DATE BETWEEN %s AND %s""",
        5: """SELECT SUM(H_AMOUNT) AS TOTAL_AMOUNT FROM HISTORY WHERE H_W_ID = %s AND H_DATE BETWEEN %s AND %s""",
        6: """SELECT H_DATE,H_AMOUNT FROM HISTORY WHERE H_W_ID = %s AND H_DATE BETWEEN %s AND %s""",
        7: """UPDATE HISTORY SET H_AMOUNT = %s, H_DATA = %s WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s""",
        8: """DELETE FROM HISTORY WHERE H_C_ID = %s AND H_C_D_ID = %s AND H_C_W_ID = %s""",
        9: """SELECT O_C_ID FROM ORDERS WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s""",
        10: """SELECT O_CARRIER_ID FROM ORDERS WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s""",
        11: """SELECT O_ID,O_ENTRY_D FROM ORDERS WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s""",
        12: """SELECT O_OL_CNT FROM ORDERS WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s""",
        13: """SELECT O_W_ID, O_D_ID FROM ORDERS WHERE O_ID = %s""",
        14: """INSERT INTO ORDERS (O_W_ID, O_D_ID, O_ID, O_C_ID, O_ENTRY_D, O_OL_CNT, O_ALL_LOCAL, O_CARRIER_ID_ID) VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s)""",
        15: """UPDATE ORDERS SET O_CARRIER_ID= %s WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s""",
        16: """DELETE FROM ORDERS WHERE O_W_ID = %s AND O_D_ID = %s AND O_ID = %s""",
        17: """SELECT OL_AMOUNT FROM ORDER_LINE WHERE OL_O_ID = %s AND OL_D_ID = %s AND OL_W_ID = %s""",
        18: """SELECT SUM(OL_QUANTITY) AS TotalQuantity FROM ORDER_LINE WHERE OL_O_ID = %s AND OL_DELIVERY_D BETWEEN %s AND %s""",
        19: """SELECT SUM(OL_AMOUNT) AS TotalSales FROM ORDER_LINE WHERE OL_DELIVERY_D BETWEEN %s AND %s""",
        20: """SELECT C_FIRST, C_LAST, C_PHONE FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s""",
        21: """SELECT C_CREDIT,C_CREDIT_LIM FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s""",
        22: """SELECT C_CREDIT_LIM, C_DISCOUNT FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s""",
        23: """SELECT C_SINCE, C_YTD_PAYMENT FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s""",
        24: """SELECT C_YTD_PAYMENT,C_PAYMENT_CNT, C_BALANCE FROM CUSTOMER WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s""",
        25: """UPDATE CUSTOMER SET C_CREDIT = %s, C_CREDIT_LIM = %s WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s""",
        26: """UPDATE CUSTOMER SET C_DISCOUNT = %s WHERE C_W_ID = %s AND C_D_ID = %s AND C_ID = %s""",
        27: """SELECT S_QUANTITY, S_ORDER_CNT FROM STOCK WHERE S_W_ID = %s AND S_I_ID = %s""",
        28: """SELECT S_I_ID, S_QUANTITY, S_REMOTE_CNT FROM STOCK WHERE S_W_ID = %s AND S_REMOTE_CNT > 0""",
        29: """SELECT S_QUANTITY, S_ORDER_CNT, S_REMOTE_CNT FROM STOCK WHERE S_I_ID = %s AND S_W_ID = %s""",
        30: """SELECT S_YTD FROM STOCK WHERE S_W_ID = %s AND S_I_ID = %s""",
        31: """SELECT I_PRICE FROM ITEM WHERE I_ID = %s""",
        32: """SELECT W_NAME, W_YTD FROM WAREHOUSE WHERE W_ID = %s""",
        33: """SELECT W_NAME, W_TAX FROM WAREHOUSE WHERE W_ID = %s""",
        34: """SELECT W_STREET_1, W_STREET_2, W_CITY, W_STATE, W_ZIP FROM WAREHOUSE WHERE W_ID = %s""",
        35: """UPDATE WAREHOUSE SET W_TAX = %s, W_YTD = %s WHERE W_ID = %s""",
        36: """SELECT D_NAME, D_YTD FROM DISTRICT WHERE D_ID = %s""",
        37: """SELECT D_NAME, D_TAX FROM DISTRICT WHERE D_ID = %s""",
        38: """SELECT D_STREET_1, D_STREET_2, D_CITY, D_STATE, D_ZIP FROM DISTRICT WHERE D_W_ID = %s AND D_ID = %s""",
        39: """UPDATE DISTRICT SET D_TAX = %s, D_YTD = %s WHERE D_ID = %s AND D_W_ID = %s"""
    },
}
