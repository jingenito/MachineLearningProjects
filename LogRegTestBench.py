import os, sys

from Regression.LogisticRegression import LogisticRegression as LogReg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#storing the root directory
root_dir = os.path.dirname(os.path.realpath('LogRegTestBench.py'))
#creating the data frame that will be used for email classification
emails = pd.read_csv(os.path.join(root_dir, 'LogRegPlotter', 'emails.csv'))

import nltk
from nltk.corpus import stopwords
#stop words are words that are regarded as unimportant in emails, such as 'the' and 'and', etc...
nltk.download('stopwords')
#saving to remove the stopwords from the data set
is_a_stopword = emails.columns.isin(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#dropping the stopwords
no_stop_words = emails.loc[:, ~is_a_stopword].copy()
#dropping the ID column
no_stop_words.drop(columns="Email No.", inplace=True)
#separating the data into X and y matrices
X = no_stop_words.drop(columns="Prediction")
y = no_stop_words['Prediction']

#split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=7, stratify=y)

######################################################################
##preprocessing stage
######################################################################
#scale the feature data so it has mean = 0 and standard deviation = 1
scaler = MinMaxScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)

#create and train the model
log_reg = LogReg()
log_reg.fit(X_train, y_train)

#score the model
from Metrics import accuracy_score as score

results = log_reg.predict(X_test)
print(score(np.array(y_test), results))

# from sklearn.linear_model import LogisticRegression

# slog_reg = LogisticRegression()
# slog_reg.fit(X_train, y_train)
# print(slog_reg.score(X_test, y_test))