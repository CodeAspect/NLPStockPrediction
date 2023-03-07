import pandas as pd
import numpy as np
import glob
import json

# path = r'C:\Users\Mark\Documents\School\CSE573\Data\Tweets'
# all_files = glob.glob(path + "\*.json")

# tweets = pd.DataFrame(columns = ["Date", "Text"])

    
# for file in all_files:
#     with open(file, "r", encoding='utf-8') as read_file:
#         data = json.load(read_file)
        

#     for d in data:
#         for t in d:
#             date = t['created_at'][:10]
#             text = t['text']
#             tweets = tweets.append({"Date":date, "Text":text}, ignore_index = True)
    
# tweets = tweets.sort_values('Date')
# print(tweets)

# tweets.to_csv(r'C:\Users\Mark\Documents\School\CSE573\News\BTCTweetData.csv', index=False)

tweetData = pd.DataFrame(columns = ["Date", "Text"])

tweets = pd.read_csv(r'C:\Users\Mark\Documents\School\CSE573\Data\twitterStock_1.csv')

for i, r in tweets.iterrows():
    tweetData = tweetData.append(pd.DataFrame({"Date":[str(r['Time'])[:10]], "Text":r['Tweet']}))
    
print(tweetData)