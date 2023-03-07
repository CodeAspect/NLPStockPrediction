import pandas as pd
from langdetect import detect 
import pickle

## Stock Qualifiers ##
def stockKeywords(text):
	if ("Apple" in text 
	 or "Amazon" in text
	 or "NYSE" in text
     or "NASDAQ" in text):
		return True
	else:
		return False
    
## List of Top 23 Companies ##
def fortune500(text):
	if ("MSFT" in text
	 or "FB"   in text
	 or "TSLA" in text
	 or "GOOG" in text
	 or "NVDA" in text
	 or "GOOG" in text
	 or "ADBE" in text
	 or "PEP"  in text
	 or "CSCO" in text
	 or "AVGO" in text
	 or "CMCSA"in text 
	 or "COST" in text 
	 or "INTC" in text 
	 or "PYPL" in text 
	 or "QCOM" in text 
	 or "NFLX" in text 
	 or "TXN"  in text 
	 or "INTU" in text 
	 or "HON"  in text 
	 or "TMUS" in text 
	 or "AMGN" in text 
	 or "AMD"  in text 
	 or "AMAT" in text 
	 or "SBUX" in text):
		return True
	else:
		return False    

rawNews = pd.read_csv(r"C:\Users\Mark\Documents\School\CSE573\News\NewsData.csv")
d2 = pd.read_csv(r"C:\Users\Mark\Documents\School\CSE573\News\cAAPLNews.csv")

news = pd.DataFrame()
for i, r in rawNews.iterrows():
    print(r)
    if stockKeywords(r['Text']) and detect(r['Text']) == "en" and not fortune500(r['Text']):
        news = news.append(r)
        
with open('news.pickle', 'wb') as handle:
    pickle.dump(news, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('news.pickle', 'rb') as handle:
    news = pickle.load(handle)
        
print(news)
        
d1 = news[news['Text'].astype(str).str.contains("aapl","AAPL")]

print(d1)
print(d2)
print(d1.shape[0])
print(d2.shape[0])
dn = pd.concat([d1, d2])

with open('dn.pickle', 'wb') as handle:
    pickle.dump(dn, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('dn.pickle', 'rb') as handle:
    dn = pickle.load(handle)
    
dn.drop_duplicates(inplace=True)

print(dn)

dn.to_csv('C:\\Users\\Mark\\Documents\\School\\CSE573\\AAPLNews.csv', index=False)