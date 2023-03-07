import pandas as pd
import numpy as np
import re


lines = []
with open(r'C:\Users\Mark\Documents\School\CSE573\Data\CurrentGMEnews.txt') as f:
    lines = f.readlines()
    
    
currentData = pd.DataFrame(columns = ["Date", "Text"])
# print(lines[0][1:-1])
split = []
for s in re.split('],',lines[0][1:-1]):
    split = s[2:-1].split('",')
    if(len(split) == 3):
        currentData = currentData.append({"Date":split[1], "Text":split[2]}, ignore_index=True)
        
currentData['Date'] = [x.replace('"','').replace(" ","") for x in currentData['Date'].values]
currentData['Text'] = [x.replace('"','') for x in currentData['Text'].values]

print(currentData['Date'].values)

currentData.to_csv(r'C:\Users\Mark\Documents\School\CSE573\News\cGMENews.csv', index=False)