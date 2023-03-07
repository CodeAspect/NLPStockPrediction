import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import Precision, Recall
import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv("NN_datasets/Final_Dataset/Stock_Final.csv")

gt = df["groundTruth"]
df.pop("groundTruth")

df = tf.convert_to_tensor(df)
gt = tf.convert_to_tensor(gt)

model = Sequential()
model.add(Dense(6542, input_dim=9764, activation='relu'))
model.add(Dense(4383, activation='relu'))
model.add(Dense(2936, activation='relu'))
model.add(Dense(1468, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall()])
print("Start fitting model")
model.fit(df, gt, epochs=100, batch_size=50, verbose=0)

'''_, accuracy, precision, recall = model.evaluate(df, gt)
print('Accuracy: %.2f' % (accuracy*100))
print('F1 Score: %.2f' % (2*((precision*recall)/(precision+recall))))
print('Recall: %.2f' % (recall))'''

predictions = (model.predict(df) > 0.5).astype(int)
print(classification_report(gt, predictions, zero_division=0))
