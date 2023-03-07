import json
import pandas as pd
import numpy as np
import glob
from datetime import datetime as dt
import pytz

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


#Extract News Data to newsMatrix DataFrame ##
news = pd.DataFrame()
fileDate = "2018"
news = pd.concat([news,buildNewsMatrix(fileDate, 13)])
fileDate = "2019"
news = pd.concat([news,buildNewsMatrix(fileDate, 3)])

news.to_csv('NewsData.csv', index=False)