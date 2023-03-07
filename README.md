# NLP Stock Prediction

Nate Berlandier 
SCAI 
Arizona State University Tempe, USA nberland@asu.edu

Shefali Deosthali 
SCAI 
Arizona State University Tempe, USA sdeostha@asu.edu

Saarthak Jain 
SCAI 
Arizona State University Tempe, USA sjain191@asu.edu

Benjamin Mixon-Baca 
SCAI 
Arizona State University Tempe, USA bmixonba@asu.edu

Mark Pop 
SCAI 
Arizona State University Tempe, USA mpop2@asu.edu

Nigel Wong 
SCAI 
Arizona State University Tempe, USA njwong2@asu.edu 
 
Abstract—The stock market is well known as one of the best ways to build wealth, however, millions of people lose money trading stock every day. Many people suggest that the market is mostly random and cannot be predicted, but we believe otherwise. Our final report outlines applications to improve trading results based on predicted outcomes of stock prices, after major news reports and tweets have been published. Our model uses these information sources to build a model that could lead to traders becoming more profitable. 
Index Terms—Sentiment Analysis, Stock Market Prediction, Word2Vec, SVM, Random Forest, Logistic Regression, Neural Networks

I. INTRODUCTION 

Our goal was to create an application that scrapes news reports and social media posts pertaining to top companies within the stock market and predict the changes in stock price for a given day. Research into this subject matter has been conducted before, but we were able to find new feature group configurations which yield better or more interesting results. Also, we used different machine learning models in hopes to get exhaustive testing scenarios and compare results from previous works. 

II. PROBLEM STATEMENT 

With so much news and buzz daily on traditional news mediums like Bloomberg, as well as on social media platforms like Twitter, the problem we are trying to solve is how can we perform stock price prediction using news articles and tweets to help investors make more informed trading decisions. Explained in the next few pages, we hope that using this model, individuals can help better predict price movement of specific securities. The metrics used to evaluate stock price prediction model performance are accuracy, F-score, and return-oninvestment (RoI) using paper trading. This system can be used to potentially improve the trading outcomes of a professional or be used as an engine for an automatic trading algorithm. 

III. RELATED WORKS 

Since the beginning of the financial markets, traders have been trying to device various strategies to beat the market. Since there is a direct element of financial gain involved, there has been a lot of work done previously in this field. Schumaker et al. [6] proposed a method for estimating the magnitude of price change by crawling 9,211 financial news articles and 10,259,042 stock quotes covering the S&P 500 stocks during a five-week period. They then extracted two feature sets: a bag-of-words and noun phrases. They applied a Support Vector Machine to the data and achieved 56% accuracy and ROI of 2.06% in a simulated trading engine. 
Ding et al. [3] applied a deep learning architecture to event vectors, which are similar to word vectors (i.e., word2vec), except are based on higher-level relationships, such as, “Microsoft sues Barnes & Noble” to capture higher-level concepts that can affect asset prices. Their approach performed 6% better than previous state-of-the-art. 
Pagolu et al. [5] utilized Sentiment Analysis and Natural Language Processing to predict stock market moment using scraped Twitter data with the concept of Word2vec. 
Chan et al. [2] collected an extensive database of public stories about companies from major news sources and analyzed the time it took after a news breakout to affect the stock price. They analyzed the data of the stocks with public news or with a price momentum in a specific month and found the stocks which had public news showed some momentum but those which didn’t have any news and had a large price movement had a price reversal. 
Mao et al. [4] measured whether the S&P 500 index direction is correlated with the number of tweets mentioning the S&P 500. They performed a stratified analysis by investigating correlations between the index, industry sector, and individual stocks and applied a linear regression model to find a strong correlation between the number of tweets and price direction. 

IV. DATASETS 

A.	Descriptions & Sizes 
We initially started with a dataset we received from the professor, which was a set of news articles about both Apple and Amazon stocks with a size of around 80k individual articles. These articles ranged from the beginning of 2018 to the beginning of 2019. We decided to scrape more current news data on specific stocks ranging from 2019 to present. We did this for Apple, Amazon, and extended the project to include GameStop since it’s been a popular stock in the last year. Apple consisted of 1,669 news articles, Amazon consisted of 1,733 news articles, and GameStop news articles consisted of 344 articles. For more articles we scraped The Guardian, a financial news website. We got more articles on Apple, Amazon, and for some novelty we added Bitcoin. In total there were 2,630 news articles that were scraped from The Guardian. Finally, we scraped Twitter for tweets pertaining to Bitcoin. We scraped 17 different twitter accounts for a total of 49,502 tweets. As for stock price charts, we obtained new charts using the free AlphaVantage API, as well as publicly available information from Yahoo Finance. The stock charts for each ticker (AAPL, AMZN, GME, BTC) were dated from 2019-02-02 to 2022/03/31. 

B.	Preprocessing Steps 
The data had to be converted from a raw JSON format extracted from the scrappers to a schema on which the machine learning models could run training and testing efficiently. The various news articles/tweets were filtered out if they didn’t contain specific key words such as the company name (Apple, Amazon, or GameStop) or exchange names (NASDAQ, NYSE, DOW, etc.) for USbased securities. Documents that were in languages other than English were removed. The remaining documents went through a process that removes new lines, special characters, punctuation, as well as non-ascii characters. Stop words such as ”a”, ”the”, ”is”, and ”are” were removed, all text was converted to lowercase, and each word was lemmatized into its root. The documents are then split into date segments where each day has a collection of documents associated with that date. 

![alt text](https://github.com/CodeAspect/NLPStockPrediction/blob/main/VISUALIZATIONS/SysArchitecture.png?raw=true)

Fig. 1. System Architecture


Ground Truth labels were created using the stock’s price data. This is accomplished by subtracting the open from the closing price (1 if the close is higher than open, 0 otherwise). The news data was split out into days and assigned the next trading day’s label. Date was normalized to be in the EST time zone so that it directly correlates with the NYSE open and close times. These were formatted into a Pandas data frame then saved to a CSV file. The idea was to get the different data formats into one consistent data template so they can all be entered into a single Python script. If the news data was released on a weekend or holiday, then the next available trading day’s label was assigned. 
News data broken out into days from the Data Engineering phase were vectorized using Sci-Kit Learn’s CountVectorizer function, which was used to create the Document Term Matrix. We used a financial sentiment dictionary, the Loughran McDonald Master Dictionary [13], that contains ten thousand words which indicate Positive, Negative, Litigious, or Uncertain sentiment. Each word in the dictionary is a feature in a new matrix. Every row in the new matrix is a zero filled vector that represents each document. If the word exists in the document, then a 1 is placed in the column representing the word, otherwise it stays 0. The corresponding label was appended to the end of the row and the matrix is exported to a CSV file to be used for testing and training a machine learning model. 

V. SYSTEM ARCHITECTURE & ALGORITHMS 

The complete system architecture is given in Figure 1. As it can be seen, the complete architecture has 4 main tasks: Data Collection, Data Pre-Processing, Classification Models and Evaluation. In the previous sections, we covered the data collection tasks and part of data pre-processing that was done on the dataset. In this section, we first describe the additional features (sentiment analysis and technical analysis) that were configured into the data and then describe the various machine learning algorithms used on this data. 

A.	Feature Engineering 

After we receive the documents (news articles and tweets), we appended two additional features which help describe the data better: sentiment analysis score and technical analysis indicators. 
Sentiment analysis is a critical component in the prediction of the future trend of the stock market. A positive sentiment would help increase the buyers’ interest in the stock and they would be willing to pay a slight premium according to the valuation they think is fair for the stock, hence increasing the price of the stock. On the other hand, a negative sentiment is detrimental to the growth of the stock price. Since the news articles and tweets articles can be unstructured (they can contain images, videos, gifs, emojis, etc.), finding the sentiment of the document is a complex task for the machines. Also, news and tweets, though serving the same purpose of helping people communicate, are very different from each other. Tweets are limited character and contain many slangs while news can range from a brief article to a detailed analysis. Hence, we resolve to using 3 different sentiment analysis scores to supplement our document. These scores are derived from NLTK’s Vader corpus, which works better with slangs and modern language, Textblob which uses unarygrams and works well with formal language and our own developed sentiment analysis model built from the McDonal Master Dictionary corpus which helps classify financial sentiments more accurately. Since a particular ticker might receive multiple news articles from various sources for the same day, we average the values over a day to predict the movement pattern of the stock for the day. 
Since the effect of a news article on the stock price can range over several days, we use some technical indicators to help identify the patterns that emerge on the stock charts. The main indicators that we used were trading volume, moving average convergence divergence (MACD), which reveals the changes in the strength, direction, momentum, and duration of a trend in a stock’s price, Relative Strength Index (RSI), which charts the current and historical strength or weakness of a stock or market based on the closing prices of a recent trading period, Bollinger Bands, a volatility indicator that measures the relative high or low of a security’s price in relation to previous trades, and Average True Range (ATR), which is also a volatility indicator but more based on the moving average of recent trades. 
Finally, the document term matrix received after the pre-processing step is added with the 2 additional features received from this feature engineering step to formulate the data that the classification models are built on. 

B. Classification Models 

Since the prediction of the stock market price is a classic classification problem, we used the following machine learning models to learn the trends in the data and to predict the accuracy with which we can predicts the movement of the ticker. 

5.2.1. Random Forest - Random Forest is an ensemble learning method that constructs multiple decision trees on the training model and the output of the random forest is the class selected by most trees. They can generate reasonable predictions with little configurations and helps prevent over-fitting on the training data (outperforming decision trees). We used the random forest classifier provided by the sci-kit learn library. In the Sci-kit classifier, multiple decision trees are created, and they each fit a number of decision tree classifiers on various sub-samples of the data set and uses averaging to improve the predictive accuracy and like mentioned above, control over-fitting. As for the hyper-parameters going into the classifier, we utilized the Sci-kit defaults. 

5.2.2. Logistic Regression - Logistic regression is a statistical technique of modelling the training data to our binary classes. We used 3 variations of logistic regression: classic logistic regression, sparse logistic regression, and logistic regression CV, all with the help of sci-kit learn library. Since the data contains a lot of binary columns (contributed by the 10k features from the Document Term Matrix), we expected and observed sparse logistic regression model to perform the best, which basically helps find the correlations in sparse data well. 

5.2.3. Neural Networks - Neural networks consist of an artificial network of functions, called parameters or neurons, to emulate the learning process occurring in human brains. Each neuron produces an output after receiving one or multiple inputs and its output is propagated to the next layer of the neural networks, which use them as inputs of their own function, and produce further outputs. This continues until every layer of neurons have been considered, and the terminal neurons have received their input. Those terminal neurons then output the result for the model. We can further convert our neural network to a reinforcement learning network which fine tunes the parameters of the model based on the new data that it sees. Figure 2 depicts an example neural network architecture similar to the one we used in our experiments. Using the python library TensorFlow, our neural network is a sequential network that has every 
 
![alt text](https://github.com/CodeAspect/NLPStockPrediction/blob/main/VISUALIZATIONS/NNArcitecture.png?raw=true)

Fig. 2. Neural Network Architecture

layer take in an input size smaller than the feature array or output from another layer. Our network consists of five layers: one input, three hidden and one output layer. Each layer, except for the output layer, used the ReLu function for evaluation, where the output layer used the sigmoid. These functions were most desirable since these two both predict on a 0 or 1 basis, where sigmoid is used as a squashing function to predict with a set of real numbers 

VI. EVALUATION 

We evaluated the models by initially splitting the data into 80% training and 20% for final model validation. 10-fold cross-validation was then be applied to the 80% training data which yielded the model with highest F-score. This final model was then be applied to the remaining 20% validation data for our final model evaluation. We used three metrics for performance evaluation in stock market directional prediction: accuracy, F-score, and ROI from paper trading. We used these three and additionally we further decomposed the accuracy metrics based on positive and negative changes, respectively. We have not observed such a decomposition in the literature when evaluating correlations between textual data and stock price movements. Our hypothesis is that text will affect both positive and negative estimations equally, hence, it will be surprising if we observe deviations e.g., where text more accurately predicts price increase. 
Table I summarizes the results of the three models we used for predictions. We found that the neural network consistently predicted stock price direction most accurately with an average accuracy of 60%. Model Accuracy F-Score Recall Neural Network 0.58 0.43 0.58 Sparse Logistic Regression 0.53 0.51 0.53 Random Forest 0.56 0.54 0.55 
 
![alt text](https://github.com/CodeAspect/NLPStockPrediction/blob/main/VISUALIZATIONS/ROCCurve.png?raw=true)

Fig. 3. Random Forest RoC curve

![alt text](https://github.com/CodeAspect/NLPStockPrediction/blob/main/VISUALIZATIONS/ModelEval.png?raw=true)
 
MODEL EVALUATION METRICS. ACCURACY EVALUATED USING 10-FOLD CROSS VALIDATION. F-SCORE, AND RECALL

Next, we analyzed the performance of our classifiers using the receiver-operator characteristic (RoC) curve. RoC curves are useful for determining how well a binary classifier is at discriminating true positives from the rest of the data set. We found that the random forest classifier in Figure 3 appeared to have high recall even for small numbers of samples and maintained a 100% accuracy which indicates biasing or over-fitting. By contrast, the neural network RoC curve in Figure 4 had what one might considered a “normal” RoC curve, indicating that the model is not over-fitting and the data are not biased. We further analyzed the model behavior using learning curves. 
We found that the neural network learning curve (Figure 5 had training and validation curves that converge to the same value, although the space between the two indicates that the model is still somewhat over-fitting to the data. This could be mitigated with a combination of more data and features useful for discriminating positive and negative samples. By contrast, the extra trees learning curve in Figure 6 (which is the same as for the random forest) contains a significant amount of over-fitting as evidenced by the large disparity between training and validation accuracy scores. 
 
![alt text](https://github.com/CodeAspect/NLPStockPrediction/blob/main/VISUALIZATIONS/NNROC.png)

Fig. 4. Neural network RoC curve

![alt text](https://github.com/CodeAspect/NLPStockPrediction/blob/main/VISUALIZATIONS/NN_LC.png?raw=true)
 
Fig. 5. Neural Network Learning Curve

![alt text](https://github.com/CodeAspect/NLPStockPrediction/blob/main/VISUALIZATIONS/ET_LC.png?raw=true)
 
Fig. 6. Extra Trees Learning Curve

VII. UI VISUALIZATION DESIGNS 

A lot of our UI visualization designs are from our paper trading analysis, which used a combination of technical analysis, semantic, and direction predictions in combination with cross-over analysis to generate buy and sell signals for the remaining 20% of the data held out. The first strategy use a 20-day exponentially weighted moving average (EMA-20) and the price to generate buy and sell signals. If the EMA-20 crossed above the price, the signal was buy and if it was below, the signal was sell. This was represented as a vector of 0 for sell and 1 for buy. We found this had an ROI of 175%. Figure 7 depicts these results. 
Next, we used the sentiment of the text data along with the EMA-20 signal. Specifically, used a component-wise multiplication of the EMA-20 buy, sell vector with the daily sentiment of 0 (negative sentiment) or 1 (positive sentiment). We found the ROI to be 120%. Figure 8 depicts these results. 
Finally, we used the EMA-20 and direction predictions to generate buy and sell signals. We again perform a component-wise multiplication of the EMA-20 and direction prediction vectors to generate buy and sell signals.
 
![alt text](https://github.com/CodeAspect/NLPStockPrediction/blob/main/VISUALIZATIONS/PT1.png?raw=true)

Fig. 7. Paper trading using 20-MA-Price cross-over strategy

![alt text](https://github.com/CodeAspect/NLPStockPrediction/blob/main/VISUALIZATIONS/PT2.png?raw=true)
 
Fig. 8. Paper trading using 20-MA-Price * Sentiment cross-over strategy

We found the ROI to be 60%. Figure 9 depicts these results. 
Not pictured is the buy and hold strategy. We found that the buy and hold strategy to produce the highest ROI, followed by the EMA-20 strategy, the EMA-20 * Sentiment, and last was EMA-20 * Direction prediction. 
 
![alt text](https://github.com/CodeAspect/NLPStockPrediction/blob/main/VISUALIZATIONS/PT3.png?raw=true)

Fig. 9. Paper trading using 20-MA-Price * Direction Prediction crossover strategy

VIII. TIMELINE AND TASK DIVISION 

Table II provides a summary of our task breakdown, the group member(s) assigned to the given task, and the date on which it started. 
To begin the project, everyone was tasked with performing a Literature Review on the topics related to the project. This included reading journals, articles, web pages, and all forms of resources that could help us understand the project and provide our solutions to the problem at hand. This task was due on the 20th of January 2022 and involved all members of the team. 
Nigel, Mark, Nate and Shefali gathered the stock price charts and news/tweets data from various sources, which helped formulate the preliminary data set by 1st Feb. Tweet and news scrappers were also deployed on cloud platforms which were scheduled to run once a day to gather the documents of interest for the day, which ran daily until 10th March. 
The project proposal was a task all group members of the team contributed to. We ensured that our preliminary data sourcing was complete before starting. Each team member was assigned a specific section to complete, and all members agreed about the contents of the proposal. 
In the meantime, Saarthak and Benjamin undertook the task of doing a thorough literature survey to understand the current trends of the field and hypothesizing the potential machine learning models that could improve upon the current benchmarks achieved in the field. 
Nigel and Benjamin worked together to add cryptocurrency data to our data set as one of our project extensions. They both researched publicly available Bitcoin price information databases and landed on Yahoo Finance as a trusted source. They collected Bitcoin price data and performed Data Engineering on the data set. 
For finalizing scraped data - Benjamin, Nate, and Nigel checked on all our existing data sets to make sure everything was in place and complete before the project moved on into its final stages. 
Once the data set was finalized, Mark and Saarthak worked on feature engineering tasks. Mark appended the data set with a document term matrix built from the McDonald Master Financial Dictionary and Saarthak performed sentiment analysis. Benjamin added additional features regarding the technical analysis of the stocks and Saarthak compiled the final data for the machine learning tasks. 
Model Evaluation involved everyone in the team as we split into 3 teams of 2 to implement and evaluate the machine learning models that we had decided on for this project. Benjamin and Nathan worked on the Neural Network, Nigel and Shefali worked on the Random Forest Classifier, and Saarthak and Mark worked on the sparse logistic regression. After all testing had completed, each team presented their evaluation findings in a group meeting. 
For Presentation Slides, we collectively contributed to a shared Google slides presentation. Each member contributed a write-up into each area of their own expertise, as well as performed peer reviews into each other’s material. Nigel provided the overall look of the presentation. 
Finally, for the Final Report, everyone contributed once again to a write-up of their areas of expertise. Collectively, we met once again to proofread the report and submit the paper by the deadline. 
Shown below is a summary of all the major tasks in tabular form for simplicity. 

![alt text](https://github.com/CodeAspect/NLPStockPrediction/blob/main/VISUALIZATIONS/TasksDone.png?raw=true)
 
IX. FUTURE WORK 

Because of the dynamic nature of the stock market, many of the existing models do not predict the movement of a stock to a high accuracy. We believe this is because most of the current models depend on the assumption that terms mentioned in news, tweets, etc. are correlated with price movement, yet this is not necessarily the case as evidenced by works achieving performance equivalent chance. We found that technical analysis of the ticker will be heavily correlated with price movements and did increase model performance compared to sentiment-only features. Also, most of the current models overlook the effect of external factors like the change in economic policies or major events like a pandemic or war. Hence, we believe that incorporating sentiment of peripheral, yet impactful, events will lead to more accurate predictions and paper-trade our model to see how well it performs. 
Future researchers should reverse engineer part of the model to extract the twitter handles and news sources that predicted the movement of a ticker to a higher accuracy. Finally, we believe using the dictionary [5] [1] will further help future researchers refine models using reinforcement learning to assign more weight to the predictions made by these sources. 
Finally, for paper trading, we explored novel combinations of technical features (EMA-20), sentiment features, and price direction predictions using a crossover trading strategy. Future researchers should explore other ways of combining the data to generate buy and sell signals, such as addition, or other arithmetic operations derived from the different vectors. 

X. CONCLUSION 

Overall, this project taught us the dynamic nature of the stock market and the challenges which pertain to predicting the movement of a stock price chart. Although we understand the stock market can be complicated to understand at times, we found that it is possible to predict the outcomes of stock prices through data with high accuracy. We extracted key features from news reports and social media and used different machine learning models to compare results. We evaluated the models based on the accuracy, F-score, and paper trading. 

REFERENCES 

[1] Hana Alostad and Hasan Davulcu. Directional prediction of stock prices using breaking news on twitter. Web Intelligence and Agent Systems: An International Journal 5, pages 1–18, 2016. 
[2] Wesley S. Chan. Stock price reaction to news and no-news: drift and reversal after headlines. Journal of Financial Economics, 70(2):223–260, 2003. 
[3] Xiao Ding, Yue Zhang, Ting Liu, and Junwen Duan. Deep learning for event-driven stock prediction. In Proceedings of the 24th International Conference on Artificial Intelligence, IJCAI’15, page 2327–2333. AAAI Press, 2015. 
[4] Yuexin Mao, Wei Wei, Bing Wang, and Benyuan Liu. Correlating s&p 500 stocks with twitter data. HotSocial ’12, page 69–72, New York, NY, USA, 2012. Association for Computing Machinery. 
[5] Venkata Sasank Pagolu, Kamal Nayan Reddy Challa, Ganapati Panda, and Babita Majhi. Sentiment analysis of twitter data for predicting stock market movements. International conference on Signal Processing, Communication, Power and Embedded System (SCOPES), pages 1–6, 2016. 
[6] Robert P. Schumaker and Hsinchun Chen. Textual analysis of stock market prediction using breaking financial news: The azfin text system. ACM Trans. Inf. Syst., 27(2), mar 2009.
