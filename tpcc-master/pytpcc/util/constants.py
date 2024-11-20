zipF_table = {
    "SalesMan":      {"a": 3, "N": 8},
    "WarehousesMan": {"a": 2, "N": 7},
    "FinancialMan":  {"a": 1, "N": 8}
}

zipF_sql = {
    "SalesMan": {
        "CUSTOMER":   {"a": 3, "N": 11},
        "ORDERS":     {"a": 3, "N": 8},
        "ORDER_LINE": {"a": 3, "N": 6},
        "HISTORY":    {"a": 3, "N": 6},
        "ITEM":       {"a": 3, "N": 3},
        "STOCK":      {"a": 3, "N": 3},
        "WAREHOUSE":  {"a": 3, "N": 2},
        "DISTRICT":   {"a": 3, "N": 2}
    },
    "WarehousesMan": {
        "STOCK":      {"a": 3, "N": 9},
        "WAREHOUSE":  {"a": 3, "N": 6},
        "ORDERS":     {"a": 3, "N": 4},
        "ORDER_LINE": {"a": 3, "N": 3},
        "DISTRICT":   {"a": 3, "N": 7},
        "ITEM":       {"a": 3, "N": 4},
        "CUSTOMER":   {"a": 3, "N": 1}
    },
    "FinancialMan": {
        "HISTORY":    {"a": 3, "N": 8},
        "ORDERS":     {"a": 3, "N": 8},
        "ORDER_LINE": {"a": 3, "N": 3},
        "CUSTOMER":   {"a": 3, "N": 7},
        "STOCK":      {"a": 3, "N": 4},
        "ITEM":       {"a": 3, "N": 1},
        "WAREHOUSE":  {"a": 3, "N": 4},
        "DISTRICT":   {"a": 3, "N": 4}
    }
}

rows_table = {
    "CUSTOMER": 300000,
    "DISTRICT": 100,
    "HISTORY": 300000,
    "ITEM": 1000000,
    "ORDERS": 300000,
    "ORDER_LINE": 3000000,
    "STOCK": 1000000,
    "WAREHOUSE":  10,
}

columns_tables = {
    "CUSTOMER":   {
        "C_ID": [1,3000],
        "C_D_ID": [1,10],
        "C_W_ID": [1,10],
        "C_FIRST": [],
        "C_MIDDLE": [],
        "C_LAST": [],
        "C_STREET_1": [],
        "C_STREET_2": [],
        "C_CITY": [],
        "C_STATE": [],
        "C_ZIP": [],
        "C_PHONE": [],
        "C_SINCE": [],
        "C_CREDIT": [],
        "C_CREDIT_LIM": ['FLOAT'],
        "C_DISCOUNT": ['FLOAT'],
        "C_BALANCE": ['FLOAT'],
        "C_YTD_PAYMENT": ['FLOAT'],
        "C_PAYMENT_CNT": ['INTEGER'],
        "C_DELIVERY_CNT": ['INTEGER'],
        "C_DATA": []
    },
    "DISTRICT":   {
        "D_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "D_W_ID": ['SMALLINT'],
        "D_NAME": [],
        "D_STREET_1": [],
        "D_STREET_2": [],
        "D_CITY": [],
        "D_STATE": [],
        "D_ZIP": [],
        "D_TAX": ['FLOAT'],
        "D_YTD": ['FLOAT'],
        "D_NEXT_O_ID": ['int'],
    },
    "HISTORY":    {
        "H_C_ID": ['int'],
        "H_C_D_ID": ['int'],
        "H_C_W_ID": ['int'],
        "H_D_ID": ['int'],
        "H_W_ID": ['int'],
        "H_DATE": ['date'],
        "H_AMOUNT": ['float'],
        "H_DATA": []
    },
    "ITEM":       {
        "I_ID": ['int'],
        "I_IM_ID": ['int'],
        "I_NAME": [],
        "I_PRICE": ['int'],
        "I_DATA": []
    },
    "ORDERS":     {
        "O_W_ID": ['int'],
        "O_D_ID": ['int'],
        "O_ID": ['int'],
        "O_C_ID": ['int'],
        "O_ENTRY_D": [],
        "O_OL_CNT": ['int'],
        "O_ALL_LOCAL": ['int'],
        "O_CARRIER_ID": ['int']
    },
    "ORDER_LINE": {
        "OL_O_ID": ['int'],
        "OL_D_ID": ['int'],
        "OL_W_ID": ['int'],
        "OL_NUMBER": ['int'],
        "OL_I_ID": ['int'],
        "OL_SUPPLY_W_ID": ['int'],
        "OL_QUANTITY": [],
        "OL_AMOUNT": ['int'],
        "OL_DIST_INFO": ['int'],
        "OL_DELIVERY_D": []
    },
    "STOCK":      {
        "S_W_ID": ['int'],
        "S_I_ID": ['int'],
        "S_QUANTITY": ['int'],
        "S_YTD": ['int'],
        "S_ORDER_CNT": ['int'],
        "S_REMOTE_CNT": ['int'],
        "S_DATA": [],
        "S_DIST_01": [],
        "S_DIST_02": [],
        "S_DIST_03": [],
        "S_DIST_04": [],
        "S_DIST_05": [],
        "S_DIST_06": [],
        "S_DIST_07": [],
        "S_DIST_08": [],
        "S_DIST_09": [],
        "S_DIST_10": []
    },
    "WAREHOUSE":  {
        "W_ID": [1,2,3,4,5,6,7,8,9,10],
        "W_NAME": ['fbhjjkmr', 'hbqthvdz', 'lcdekrllqm', 'mbsbelppnv', 'odldvlt', 'sgabrjfexa', 'syolzcj', 'uyyqmi', 'wqkyhypcd', 'xxprbakw'],
        "W_STREET_1": ['bdwdktwudak', 'bxctxziiaknqdar', 'dyrxvmjjnvfq', 'jbepmgjclg', 'ldmmiicstg',
                       'mobtxhyvnantis', 'mwcauhqdbqljfbrh', 'qgscqrmmfgdv', 'ubcuciwhhdubq', 'wzghramfeshcozdsy'],
        "W_STREET_2": ['ctjithvbupbov', 'dahmwnbhjthgayzj', 'dapnzgsxezbbxnyyx', 'eohcafzbfudxdynazkjl', 'jsxfairznvzordgvpa',
                       'losiyebgonrlby', 'losrjjdwqwlltg', 'rccwpmvjwmyjvtajtp', 'sdtdotcfrcnhul', 'xgfxqauwvapmihznvs'],
        "W_CITY": ['bmcwnbfqkqqvcvahnl', 'fcirmgxvktzeqswy', 'fuccnxocomp', 'hmkpwmajblynsib', 'htkpecnmkejmlkacsgwb', 'pqcxpxubiretlrbj',
                   'vvsdeuhyznmtw', 'wzspisdfbaebkumt', 'yjtxdjrtgakgjklrlbi', 'zujootbpzrucnxfijsah'],
        "W_STATE": ['cj', 'dr', 'mz', 'nv', 'rw', 'tn', 'ul', 'vn', 'yi', 'yz'],
        "W_ZIP": ['116511111', '145711111', '213411111', '244811111', '423811111', '608211111', '624211111', '746111111', '850311111', '967011111'],
        "W_TAX": [0.0033, 0.0436, 0.0531, 0.1051, 0.1082, 0.1181, 0.1193, 0.1286, 0.143, 0.1995],
        "W_YTD": [300000],
    },
}