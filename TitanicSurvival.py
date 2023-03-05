import os, sys

from Regression.LogisticRegression import LogisticRegression as LogReg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#storing the root directory
root_dir = os.path.dirname(os.path.realpath('LogRegTestBench.py'))
#creating the data frame that will be used for email classification
passengers = pd.read_csv(os.path.join(root_dir, 'LogRegPlotter', 'titanic.csv'))

# Update sex column to numerical
passengers['Sex'].replace(['female','male'],[1, 0], inplace=True)
# Fill the nan values in the age column
mean = passengers['Age'].mean()
passengers['Age'].replace([np.nan],[mean], inplace=True)

# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].copy()
passengers['FirstClass'].replace([2,3],[0,0],inplace=True)
# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].copy()
passengers['SecondClass'].replace([1,2,3],[0,1,0],inplace=True)

#################################################################################
# Select the desired features
features = passengers[["Sex","Age","FirstClass", "SecondClass","Fare"]]
labels = passengers[["Survived"]]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Perform train, test, split
training_data, validation_data, training_labels, validation_labels = train_test_split(features, labels)
# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
validation_data = scaler.transform(validation_data)

print(training_data)

# Create and train the model
log_reg = LogReg()
log_reg.fit(training_data,training_labels)

# Score the model on the train data
print('Training Score: %d' % log_reg.score(training_data,training_labels))
# Score the model on the validation data
print('Validation Score: %d' % log_reg.score(validation_data,validation_labels))