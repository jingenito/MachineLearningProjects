from LinearRegression import LinearRegression as LinReg
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv('SOCR-HeightWeight.csv')
heights, weights = df['Height(Inches)'], df['Weight(Pounds)']

line_fitter = LinReg()

start = time.time()
line_fitter.fit(heights, weights)
end = time.time()
duration = end - start
print('Data was fitted to model in %d seconds' % duration)

weights_predicted = line_fitter.predict(heights)

plt.plot(heights, weights, 'o')
plt.plot(heights, weights_predicted, '-o')

plt.show()