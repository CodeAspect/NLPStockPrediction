import pandas as pd
import numpy as np
import glob
import json

path = r'C:\Users\Mark\Documents\School\CSE573\Data\CurrentNews'
all_files = glob.glob(path + "\*.json")

news = pd.DataFrame(columns = ["Date", "Text"])

for file in all_files:
    with open(file, "r", encoding='utf-8') as read_file:
        data = json.load(read_file)
        
    date = data[0]['data'][:10]
    text = data[0]['text']
    
    news = news.append({"Date":date, "Text":text}, ignore_index = True)
    
news = news.sort_values('Date')
print(news['Date'].values)

# news.to_csv(r'C:\Users\Mark\Documents\School\CSE573\News\GuardianNewsData.csv', index=False)