from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#reading from CSV data
csv_filename = "C:\\Users\\Joseph\Source\\MachineLearningProjects\\LinearRegPlotter\\SOCR-HeightWeight.csv"
print('Reading %s' % csv_filename)
df = pd.read_csv(csv_filename)
X, Y = df['Height(Inches)'], df['Weight(Pounds)']

X = X.values.reshape(-1,1)
Y = Y.values.reshape(-1,1)

print('Fitting model to data')
line_fitter = LinearRegression()
line_fitter.fit(X, Y)
y_predict = line_fitter.predict(X)

print('Plotting data')
plt.scatter(X, Y, marker='.')
plt.plot(X, y_predict, marker='.', color='g')
plt.show()