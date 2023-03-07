import requests
import json
from os.path import exists
from datetime import datetime as dt
from datetime import date
from datetime import timedelta

def grab_stocks():
    cutoff = dt.strptime("2022-01-01", "%Y-%m-%d")

    for i in range(2):
        if i == 0:
            stock = '_apple'
            url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey=ID7KLMK65L6JXPQ1'
        else:
            stock = '_amazon'
            url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AMZN&apikey=ID7KLMK65L6JXPQ1'
    
        r = requests.get(url)
        data = r.json()
        if not exists('current_stocks_'+stock+'.txt'):
            f = open('current_stocks'+stock+'.txt', 'w')

            #stack = []
            for i in list(sorted(data['Time Series (Daily)'].keys())):
                iDate = dt.strptime(i, "%Y-%m-%d")
                if iDate >= cutoff:
                    temp = {i: data['Time Series (Daily)'][i]}
                    f.write(json.dumps(temp))
            
        else:
            f = open('current_stocks'+stock+'.txt', 'a')
            today = date.today()
            yesterday = today - timedelta(days = 1)
            yesterday = yesterday.strftime("%Y-%m-%d")
        
            temp = {yesterday: data['Time Series (Daily)'][yesterday]}
            f.write(json.dumps(temp))


        f.close()

def get_stock_data(sym='AAPL'):
    cutoff = dt.strptime("2010-01-01", "%Y-%m-%d")
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey=ID7KLMK65L6JXPQ1'.format(sym)
    r = requests.get(url)
    data = r.json()
    if not exists('current_stocks_'+sym+'.txt'):
        f = open('current_stocks_'+sym+'.txt', 'w')

        #stack = []
        for i in list(sorted(data['Time Series (Daily)'].keys())):
            iDate = dt.strptime(i, "%Y-%m-%d")
            if iDate >= cutoff:
                temp = {i: data['Time Series (Daily)'][i]}
                f.write(json.dumps(temp))
            
    else:
        f = open('current_stocks'+sym+'.txt', 'a')
        today = date.today()
        yesterday = today - timedelta(days = 1)
        yesterday = yesterday.strftime("%Y-%m-%d")
        
        temp = {yesterday: data['Time Series (Daily)'][yesterday]}
        f.write(json.dumps(temp))
    f.close()


if __name__ == '__main__':
    get_stock_data()

