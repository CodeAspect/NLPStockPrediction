import pandas as pd

AMZN_Labels = pd.read_csv(r'CHARTS/AMAZON1440.csv')

AMZN_Labels.columns = ['Date','Time','Open','High','Low','Close','X']
AMZN_Labels = AMZN_Labels[['Date','Open','Close']].copy()
AMZN_Labels['Date'] = AMZN_Labels['Date'].str.replace('.', '-', regex=True)

AMZN_Labels.to_csv('AMZN_Labels.csv', index=False)