import json
import pandas as pd
import numpy as np
import pickle
import re
import glob
from datetime import datetime as dt
import pytz
from langdetect import detect 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer


## Normalize News Dates/Times ##
def normalizeDateTime(dateTime):
	currentDateTime = dt.fromisoformat(dateTime)

	curTZ = pytz.timezone("EET")
	newTZ = pytz.timezone("US/Eastern")

	localizedTZ = curTZ.localize(currentDateTime)
	newTZTimestamp = localizedTZ.astimezone(newTZ)

	newDateTime = newTZTimestamp.strftime("%Y-%m-%d %H:%M:%S")

	return newDateTime


## Extract Relevant Data From JSON Files ##
def extractJSONData(fileDate):
	news = pd.DataFrame(columns = ["Date", "Time", "Text"])

	path = r'/home/icon/NEWS/'+fileDate+'_d157b48c57be246ec7dd80e7af4388a2'
	all_files = glob.glob(path + "/*.json")

	for file in all_files:

		with open(file, "r") as read_file:
			data = json.load(read_file)

		newDateTime = normalizeDateTime(data["published"][:19])

		text = data["text"]
		date = newDateTime[0:10:1]
		time = newDateTime[11:19:1]

		row = pd.DataFrame({"Date": [date], "Time": [time], "Text": [text]})
		news = pd.concat([news,row])

	news = news.sort_values(['Date','Time'])

	return news

## Build The News Matrix ##
def buildNewsMatrix(fDate, end):
	news = pd.DataFrame()
	for i in range(1, end):
		fileMo = ""
		fileDate = fDate
		fileMo = "0" + str(i)
		fileMo = fileMo[-2:]
		fileDate = fileDate + '_' + fileMo
		news = pd.concat([news,extractJSONData(fileDate)])

	return news

## Stock Qualifiers ##
def stockQualifier(text):
	if ("NYSE" in text 
	 or "NASDAQ" in text
	 or "Apple" in text
	 or "Amazon" in text):
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

## Lemmatize Document ##
def docLem(r, lem):
	tempDoc = [lem.lemmatize(word) for word in r['Text'].split(" ")]
	return [" ".join(word) for word in [tempDoc]][0]

## Clean News ## 
def cleanNews(news, re):
	lem = WordNetLemmatizer()
	regex = re.compile('[^a-zA-Z ]')
	tempNews = []

	for i, r in news.iterrows():
		if stockQualifier(r['Text']) and detect(r['Text']) == "en" and not fortune500(r['Text']):
			r['Text'] = r['Text'].replace('\n','').replace(':',' ').replace('.',' ').replace('-',' ').lower()
			r['Text'] = regex.sub('', r['Text'])
			r['Text'] = docLem(r, lem)
			tempNews.append(r.values)
			print(r)

	stockNews = pd.DataFrame(tempNews, columns = ["Date", "Time", "Text"])
	stockNews.drop_duplicates(inplace=True)

	return stockNews

## Split By Ticker And Arrange Into Sections By Date ## 
def splitByTicker(cleanedNews, ticker):
	newsMatrix = []

	tNews = cleanedNews[cleanedNews['Text'].astype(str).str.contains(ticker)]

	newsDates = tNews['Date'].unique()
	for date in newsDates:
		newsMatrix.append(tNews[tNews['Date'] == date])
	  
	return newsMatrix

## Create Document Term Matrix With Finaincial Sentiment Dictionary ##
def createDocumentTermMatrix(newsMatrix, FinancialDict):
	docTermMatrix = pd.DataFrame(columns=FinancialDict['word'].to_list())

	for i in range(len(newsMatrix)):
		token = RegexpTokenizer(r'[a-zA-Z]+')
		cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
		text_counts= cv.fit_transform(newsMatrix[i]['Text'].values)

		docMat = pd.DataFrame(text_counts.todense())
		docMat.columns = cv.get_feature_names_out()

		docTermMatrix.loc[len(docTermMatrix)] = 0

		for col in docMat.columns:
			if col in FinancialDict['word'].to_list():
				docTermMatrix.iloc[i][col] += 1

	docTermMatrix = docTermMatrix[docTermMatrix.columns[docTermMatrix.sum(axis=0) >= 1]]

	return docTermMatrix



## MAIN ##

#Extract News Data to newsMatrix DataFrame ##
news = pd.DataFrame()
fileDate = "2018"
news = pd.concat([news,buildNewsMatrix(fileDate, 13)])
fileDate = "2019"
news = pd.concat([news,buildNewsMatrix(fileDate, 3)])

## Clean Data - Remove Irrelevant Data, Remove Numbers/Symbols, Lemmatize ##
cleanedNews = cleanNews(news, re)

## Split By Ticker And Date ##
AMZN_NewsMatrix = splitByTicker(cleanedNews, 'amzn')
AAPL_NewsMatrix = splitByTicker(cleanedNews, 'aapl')

FinancialDict = pd.read_csv(r'LM-SA-2020.csv')

## Create Document Term Matrix ##
AMZN_DTM = createDocumentTermMatrix(AMZN_NewsMatrix, FinancialDict)
AAPL_DTM = createDocumentTermMatrix(AAPL_NewsMatrix, FinancialDict)

pickle.dump(AMZN_DTM, open('AMZN_DTM.sav','wb'))
pickle.dump(AAPL_DTM, open('AAPL_DTM.sav','wb'))

AMZN_DTM = pickle.load(open('AMZN_DTM.sav', 'rb'))
AAPL_DTM = pickle.load(open('AAPL_DTM.sav', 'rb'))