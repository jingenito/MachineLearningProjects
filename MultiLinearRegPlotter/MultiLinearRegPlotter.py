from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#reading from CSV data
csv_filename = "C:\\Users\\Joseph\\Source\\MachineLearningProjects\\MultiLinearRegPlotter\\2023 QS World University Rankings.csv"
print('Reading %s' % csv_filename)
df = pd.DataFrame(pd.read_csv(csv_filename))
X, Y = df[['ar score', 'er score']], df[['Rank']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                                    train_size = 0.8, 
                                                    test_size = 0.2, 
                                                    random_state=6)

print('Fitting model to data')
ols = LinearRegression()
ols.fit(x_train, y_train)

#plot the results
print('Plotting data')
fig = plt.figure()
ax = plt.axes(projection ='3d')

ax.set_xlabel('AR Score')
ax.set_ylabel('ER Score')
ax.set_zlabel('Rank')

ax.scatter(x_train[['ar score']], x_train[['er score']], y_train, c='k', marker='+')

ax.plot_surface(np.array([[0, 0], [100, 100]]), 
                np.array([[0, 100], [0, 100]]), 
                ols.predict(np.array([[0, 0, 100, 100], [0, 100, 0, 100]]).T).
                reshape((2, 2)), alpha=.7)

plt.show()

## Creating dataset
#x = np.outer(np.linspace(-3, 3, 32), np.ones(32))
#y = x.copy().T # transpose
#z = ( 2 + (3 * x)  + (4 * y) )
 
## Creating figure
#fig = plt.figure(figsize =(14, 9))
#ax = plt.axes(projection ='3d')

## Creating plot
#ax.plot_surface(x, y, z)

## show plot
#plt.show()