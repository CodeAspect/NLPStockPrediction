import pandas as pd
import numpy as np 

df = pd.read_json(r'C:\Users\Mark\Documents\School\CSE573\Data\CHARTS\GMEdaily.json')

df = df.rename_axis('Date').reset_index(level=0)
df = df[['Date','open','close']].copy()
df.columns = ['Date','Open','Close']
df = df[::-1]

df.to_csv(r'C:\Users\Mark\Documents\School\CSE573\cGME_Labels.csv', index=False)