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
#need to remove the stopwords from the data set
is_a_stopword = emails.columns.isin(stopwords.words('english'))
emails.columns[is_a_stopword]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


