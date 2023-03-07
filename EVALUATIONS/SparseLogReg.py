import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

## Uploading Data ##
rawStockData = pd.read_csv(r'C:\Users\Mark\Desktop\Stock_Final.csv')
rawBTCTweetData = pd.read_csv(r'C:\Users\Mark\Desktop\BTCTweetData_Final.csv')
rawBTCData = pd.read_csv(r'C:\Users\Mark\Desktop\GuardianBTCNews_Final.csv')

## Combining Data ##
combined = pd.concat([rawStockData,rawBTCTweetData,rawBTCData])

groundTruth = combined['groundTruth']
combined = combined.drop('groundTruth', axis = 1)

## Calling Model ##
logReg = LogisticRegression(    
    penalty="l1",
    solver="liblinear",
    warm_start=True, 
    intercept_scaling=10000.0,)

score = []
accuracy = []
f1Score = []
recall = []

## 10 Fold Cross Validiation ##
kf = KFold(n_splits=10, random_state=True, shuffle=True)
kf.get_n_splits(combined)

for train_index, test_index in kf.split(combined):
    x_train, x_test = combined.iloc[train_index], combined.iloc[test_index]
    y_train, y_test = groundTruth.iloc[train_index], groundTruth.iloc[test_index]
    
    logReg.fit(x_train, y_train)
    
    y_pred = logReg.predict(x_test) 
    report = classification_report(y_test, y_pred, output_dict=True)
    
    score.append(logReg.score(x_test, y_test))
    accuracy.append(report['accuracy'])
    f1Score.append(report['macro avg']['f1-score'])
    recall.append(report['macro avg']['recall'])
    
## Displaying Results ##
print("Sparse Logistic Regression")
print("Score: ", np.mean(score))
print("Accuracy: ", np.mean(accuracy))
print("F1 Score: ", np.mean(f1Score))
print("Recall: ", np.mean(recall))
