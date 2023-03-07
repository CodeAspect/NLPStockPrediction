import pandas as pd
import numpy as np
import sys
import re
import datetime as dt
from datetime import timedelta
from langdetect import detect 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

## Stock Qualifiers ##
def stockKeywords(text):
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
		if stockKeywords(r['Text']) and detect(r['Text']) == "en" and not fortune500(r['Text']):
			r['Text'] = r['Text'].replace('\n','').replace(':',' ').replace('.',' ').replace('-',' ').lower()
			r['Text'] = regex.sub('', r['Text'])
			r['Text'] = docLem(r, lem)
			tempNews.append(r.values)

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

	return docTermMatrix

def groundTruthLabels(newsMatrix, newsLabels):
	newsDates = []
	for news in newsMatrix:
		newsDates.append(news.iloc[0]['Date'])

	newsLabels = newsLabels.loc[(newsLabels['Date'] >= newsDates[0]) & (newsLabels['Date'] <= newsDates[-1])]
	endLabelDate = newsLabels.iloc[-1]['Date']

	for i in range(len(newsMatrix)):
		if(endLabelDate == newsMatrix[i].iloc[0]['Date']):
			delIx = i

	newsMatrix = newsMatrix[:delIx]
	newsDates = newsDates[:delIx]

	labels = []
	for i, r in newsLabels.iterrows():
		if(r['Close'] - r['Open'] > 1):
			labels.append(1)
		else:
			labels.append(0)

	newsLabels = newsLabels.assign(Label = labels)

	groundTruth = []
	for date in newsDates:
		found = 0
		dtDate = dt.date.fromisoformat(date)
		while found == 0:
			dtDate += timedelta(days=1)
			for i, r in newsLabels.iterrows():
				if(r['Date'] == str(dtDate)):
					groundTruth.append(r['Label'])
					found = 1

	return newsMatrix, groundTruth



## MAIN ##

## Get Arguments ##
ticker = sys.argv[1]
dataPath = sys.argv[2]
labelPath = sys.argv[3]

## Import Data CSVs ##
news = pd.read_csv(dataPath)
newsLabels = pd.read_csv(labelPath)
FinancialDict = pd.read_csv(r'FinancialDictionary.csv')

## Clean Data - Remove Irrelevant Data, Remove Numbers/Symbols, Lemmatize ##
print('Cleaning Data...')
cleanedNews = cleanNews(news, re)

## Split By Ticker And Date ##
print('Splitting Data...')
newsMatrix = splitByTicker(cleanedNews, str(ticker).lower())

## Get Ground Truth Labels For Each Day ##
print('Extracting Ground Truth...')
newsMatrix, groundTruth = groundTruthLabels(newsMatrix, newsLabels)

## Create Document Term Matrix ##
print('Creating Document Term Matrix...')
DTM = createDocumentTermMatrix(newsMatrix, FinancialDict)
DTM['groundTruth'] = groundTruth

## Export Document Term Matrix ##
DTM.to_csv(str(ticker)+'_DTM.csv', index=False)
