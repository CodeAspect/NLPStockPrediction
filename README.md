# CSE573

## DataToDTM.py Instructions

#### Dependencies 
This script was set up in Python3, it requires the following libraries:
- Pandas
- NumPy
-	re
-	datetime
-	langdetect 
-	Sklearn
-	NLTK
-	FinancialDictionary.csv File
-	tensorflow

#### To Execute 
python3 DataToDTM.py *Ticker* *PathToNewsData* *PathToPriceData*

Example:

python3 DataToDTM.py AMZN NewsData.csv AMZN_Labels.csv

News Data Format:

Date (YYYY-MM-DD), Time, Text

Price Data Format:

Date (YYYY-MM-DD), Open, Close

## Crypto

Begin by creating a python3 virtual environment, sourcing it, and then installing the required
python3 packages. 

```bash
$ python3 -m venv /path/to/new/virtual/environment
$ source /path/to/new/virtual/environment

pip3 install -r requirements.txt

```

Now train the RL trader

```bash
python3 RLTrader.py
```
