from constants import START_DATE, END_DATE, WINDOW_SIZE, K, N
from utils.helper import convert
from collections import defaultdict

def generate_technical_data(extracted_data, stock_data):
    generated_data = {}
    data_idx = {}
    for key in extracted_data:
        data_idx[key] = 0
        generated_data[key] = []
    
    price_idx, price_data, current_price_data = 0, [], []
    current_day = convert(START_DATE)
    end_day = convert(END_DATE)
    while current_day < end_day:
        # Technical Indicators
        for key in extracted_data:
            while convert(extracted_data[key][data_idx[key]][0]) < current_day:
                data_idx[key] += 1
            generated_data[key].append([value for _, value in extracted_data[key][data_idx[key]-N:data_idx[key]]])

        # Fundamental Indicators
        while convert(stock_data[price_idx][0]) < current_day + K:
            price_idx += 1
        
        current_price_data.append(stock_data[price_idx - K][1])
        price_data.append(stock_data[price_idx][1])

        current_day += WINDOW_SIZE
    return generated_data, current_price_data, price_data

def generate_single_factor_data(extracted_fundamental):
    fundamental_idx = defaultdict(lambda: 0)
    generated_fundamental = defaultdict(lambda: [])
    current_day = convert(START_DATE)
    end_day = convert(END_DATE)
    while current_day < end_day:
        # Income Statement Indicators
        for key in extracted_fundamental:
            while convert(extracted_fundamental[key][fundamental_idx[key]][0]) < current_day:
                fundamental_idx[key] += 1
            generated_fundamental[key].append(extracted_fundamental[key][fundamental_idx[key] - K][1])
        current_day += WINDOW_SIZE
    return generated_fundamental