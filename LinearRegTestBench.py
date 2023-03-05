import os, sys

from Regression.LinearRegression import LinearRegression as LinReg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#storing the root directory
root_dir = os.path.dirname(os.path.realpath('LinearRegTestBench.py'))
#creating the data frame that will be used for height-weight prediction
df = pd.read_csv(os.path.join(root_dir, 'LinearRegPlotter', 'SOCR-HeightWeight.csv'))
heights, weights = np.array(df['Height(Inches)']).reshape((len(df), 1)), np.array(df['Weight(Pounds)'])

line_fitter = LinReg()

start = time.time()
line_fitter.fit(heights, weights)
end = time.time()
duration = end - start
print('Data was fitted to model in %d seconds' % duration)

print(line_fitter.get_params())

weights_predicted = line_fitter.predict(heights)

plt.plot(heights, weights, 'o')
plt.plot(heights, weights_predicted, '-o')

plt.show()