from LinearRegression import LinearRegression
from Util import JSONSerializer
from matplotlib import pyplot as plt
import pandas as pd

#reading from CSV data
csv_filename = "SOCR-HeightWeight.csv"
print('Reading %s' % csv_filename)
df = pd.read_csv(csv_filename)
X, Y = df['Height(Inches)'], df['Weight(Pounds)']

line_fitter = LinearRegression()
line_fitter.fit(X, Y)
y_predict = line_fitter.predict(X)

plt.scatter(X, Y, marker='o')
plt.plot(X, y_predict, '-o')
plt.show()