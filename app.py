# import all libs
from t_tech.invest import CandleInterval
import redis
import threading
from flask import Flask
import pandas as pd
import numpy as np
import time
import os
from conv_model import conv_model
from trade_utils import *

def load_env_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")

    with open(file_path, 'r') as file:
        for line in file:
            # Удаляем пробелы по краям и пропускаем пустые строки и комментарии
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Разбиваем строку по знаку '=' и обрабатываем переменные окружения
            key, value = line.split('=', 1)
            os.environ[key] = value


# Используем функцию для загрузки переменных из файла .env
load_env_file('.env')


app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379, decode_responses=True)

data_to_display = {'info': 'None </br>'}

# my token and other hyperparams
token = os.getenv("token")

conv_fit_size = 20

black_list = ['IRAO', 'HYDR', 'VTBR', 'UNAC', 'SIBN', 'GAZP', 'YNDX']
white_list = {'YNDX': 0.2,
              'TCSG': 0.5,
              'SBER': 0.01,
              'LKOH': 0.5,
              'ROSN': 0.05,
              'BSPB': 0.01,
              'MAGN': 0.005,
              'MGNT': 0.5,
              'PHOR': 1,
              'ALRS': 0.01,
              'PLZL': 0.5,
              'SNGSP': 0.005,
              'MTSS': 0.05,
              'OZON': 0.5,
              'BANE': 0.5,
              'FIVE': 0.5,
              'SNGS': 0.005,
              'TATN': 0.1,
              'PIKK': 0.1,
              'NVTK': 0.2,
              'CHMF': 0.2,
              'SMLT': 0.5,
              'UPRO': 0.001,
              'SVCB': 0.005,
              'LENT': 0.5,
              'AFKS': 0.001,
              'ASTR': 0.05,
              'RTKM': 0.01,
              'HHRU': 1,
              'POSI': 0.2}

# getting "companies" table
df = pd.read_csv('./companies_fixed.csv')
df['cap'] = df['cap'].astype(float)

# a function to make buy orders in accordance with the prediction


acc_id = get_acc_id(token)[0]


# scheduler will launch this every day



# a function to get a sequence of percantage changes for one stock
def load_percents(ticker, token, trhd=250, trhd_del=400):
    prices = []
    with Client(token) as client:
        try:
            r = client.market_data.get_candles(figi=df[df['ticker'] == ticker]['figi'].item(),
                                               from_=(datetime.datetime(datetime.datetime.now().year,
                                                                        datetime.datetime.now().month,
                                                                        datetime.datetime.now().day) - datetime.timedelta(
                                                   days=40)),
                                               to=(datetime.datetime(datetime.datetime.now().year,
                                                                     datetime.datetime.now().month,
                                                                     datetime.datetime.now().day)),
                                               interval=CandleInterval.CANDLE_INTERVAL_DAY)
            cop = pd.DataFrame(r.candles)
            for d in cop['close']:
                prices.append(dict_cast_money(d))
        except:
            return ['Skipped']

    percents = []
    if not prices:
        return np.array([]), np.array([]), np.array([]), np.array([])
    last_price = prices[0]
    for price in prices[1:]:
        if price != last_price:
            percents.append(round((price - last_price) * 100 * 100 / last_price, 2))
        last_price = price
    percents = percents[len(percents) - conv_fit_size::]
    avg_perc = [0] * conv_fit_size
    for i, elem in enumerate(percents):
        last_elem = 0
        prev_el = 0
        if i == 0:
            avg_perc[i] = round(elem, 2)
        elif i == 1:
            avg_perc[i] = round((elem + last_elem) / 2, 2)
        else:
            avg_perc[i] = round((elem + last_elem + prev_el) / 3, 2)
        if abs(elem) > trhd_del:
            return ['Exceeded']
        if avg_perc[i] > trhd:
            avg_perc[i] = trhd
        if avg_perc[i] < -trhd:
            avg_perc[i] = -trhd
        prev_el = last_elem
        last_elem = elem
    return avg_perc


def get_data_to_model(white_list, black_list, trhd=250, trhd_del=400):
    data = []
    tickers = {}
    ind = 0
    for i, ticker in enumerate(white_list):
        if ticker in black_list:
            continue
        avg_perc = load_percents(ticker, token, trhd, trhd_del)
        if avg_perc[0] == 'Skipped':
            tickers[ticker] = -1
            continue  # and give a report - something went wrong with df
        if avg_perc[0] == 'Exceeded':
            tickers[ticker] = -2
            continue  # report that we can't predict
        tickers[ticker] = ind
        ind += 1
        avg_perc = np.array(avg_perc)
        data.append(avg_perc)
    data = np.array(data)
    return data, tickers


def get_date():
    return datetime.datetime.now()






data_to_model, tiks = get_data_to_model(white_list, black_list)
last_date = get_date()
cache.set('last_date', str(last_date))
cache.set('tiks', str(tiks))
cache.set('data', str(data_to_model))

def repeater():
    global data_to_display
    while True:
        week_day = datetime.datetime.now().weekday()
        hr = datetime.datetime.now().hour
        mn = datetime.datetime.now().minute
        if get_date() > datetime.datetime.strptime(
                cache.get('last_date').replace("'", '').replace('b', '').replace('"', ''),
                '%Y-%m-%d %H:%M:%S.%f') + datetime.timedelta(days=1):
            data_to_model, tiks = get_data_to_model(white_list, black_list)
            last_date = get_date()
            cache.set('last_date', str(last_date))
            cache.set('tiks', str(tiks))
            cache.set('data', str(data_to_model))
        if week_day in [5, 6]:
            data_to_display['info'] = 'Today is a weekend </br>'
            # time.sleep(21600)
        elif hr == 14 and mn <= 5:
            # checking if we should update cache

            portfolio, opers = get_portf(acc_id, token)  # Получение портфолио и операций

            balance = cast_money(portfolio.total_amount_portfolio)
            actives = []  # список активов
            currencies = []
            for pos in portfolio.positions:
                active = portfolio_pose_todict(pos)
                if active['instrument_type'] == 'share':
                    actives.append(active)
                if active['instrument_type'] == 'currency':
                    currencies.append(active)

            preds, tiks, data_to_model = get_prediction(n_days=3)
            predictions = dict()
            for elem in tiks.keys():
                # res_str += elem + ' '  # + str(tiks[elem]) + '\n'
                if tiks[elem] in [-1, -2]:
                    continue
                predictions[df[df['ticker'] == elem]['figi'].values[0]] = preds[tiks[elem]][0]

            sell_resps = check_stocks_to_sell(df, actives, acc_id, token, predictions)
            time.sleep(300)
            for ord_resp in sell_resps:
                ord_id = ord_resp.order_id
                with Client(token) as client:
                    resp = client.orders.get_order_state(account_id=acc_id, order_id=ord_id)
                    if resp != 0:
                        cancel_time = client.orders.cancel_order(account_id=acc_id, order_id=ord_id)
            buy_resps = preds_to_orders(df, predictions, acc_id, token, currencies)
            data_to_display['info'] = "Operations done! </br>"
        time.sleep(1)


def get_prediction(trhd=250, trhd_del=400, n_days=1):  # n_days = 1, 2, 3, 4
    if n_days > 4 or n_days < 1:
        raise ValueError('n_days must be 1, 2, 3 or 4')

    # converting string to numpy
    dtm = cache.get('data')
    dtm = dtm.replace("[[", '')
    dtm = dtm.replace(']]', '')
    dtm = dtm.replace('"', "")
    dtm = dtm.replace("\n", ' ')
    dtm = dtm.replace(",", "")
    mas = dtm.split(']')
    # return str(mas)
    data_to_model = []
    for elem in mas:
        buf = []
        temp = elem.split(' ')
        for val in temp:
            val = val.replace('[', '')
            if val:
                buf.append(float(val))
        data_to_model.append(buf)
    data_to_model = np.array(data_to_model)

    # converting string to dict
    h = cache.get('tiks')
    h = h.replace("'", '').replace('{', '').replace('}', '').replace('"', '')
    h = h.split(', ')
    tiks = dict()
    for elem in h:
        temp = elem.split(': ')
        tiks[temp[0]] = int(temp[1])

    preds_1 = conv_model.predict(data_to_model)
    if n_days == 1:
        return preds_1, tiks, data_to_model
    preds_2 = preds_1
    preds_1 = preds_1.tolist()
    for day in range(n_days - 1):
        for i in range(data_to_model.shape[0]):
            for j in range(data_to_model.shape[1] - 1):
                data_to_model[i][j] = data_to_model[i][j + 1]
            data_to_model[i][data_to_model.shape[1] - 1] = preds_2[i][0]
        preds_2 = conv_model.predict(data_to_model)
        preds_2 = preds_2.tolist()
        for i in range(len(preds_1)):
            preds_1[i].append(preds_2[i][0])
    return preds_1, tiks, data_to_model


@app.route('/', methods=['GET'])
def master():
    global data_to_display

    preds, tiks, data_to_model = get_prediction(n_days=3)
    # data_to_model = get_prediction(n_days = 3)
    # return data_to_model
    res_str = data_to_display['info'] + 'Predictions: </br>'

    for elem in tiks.keys():
        res_str += elem + ' '  # + str(tiks[elem]) + '\n'
        if tiks[elem] == -1:
            res_str += 'Skipped</br>'
            continue
        if tiks[elem] == -2:
            res_str += 'Exceeded</br>'
            continue
        for num in preds[tiks[elem]]:
            res_str += str(num) + ' '
        res_str += '</br>'

    res_str += str(get_date()) + '</br>'
    res_str += 'Last data update ' + cache.get('last_date')
    return res_str


repeater_thread = threading.Thread(target=repeater)
repeater_thread.daemon = True
repeater_thread.start()

# master()
if __name__ == "__main__":
    # Запуск сервера Flask и планировщика
    app.run(debug=True)