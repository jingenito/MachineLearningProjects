from LinearRegression import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

#reading from CSV data
csv_filename = "2023 QS World University Rankings.csv"
print('Reading %s' % csv_filename)
df = pd.DataFrame(pd.read_csv(csv_filename))
X, Y = df[['ar score', 'er score']].values, df['Rank'].values

ols = LinearRegression()
ols.fit(X, Y)

#plot the results
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('AR Score')
ax.set_ylabel('ER Score')
ax.set_zlabel('Rank')

ax.scatter(X[:,0], Y[:,1], marker='o')

ax.plot_surface(np.array([[0, 0], [4500, 4500]]), 
                np.array([[0, 140], [0, 140]]), 
                ols.predict(np.array([[0, 0, 100, 100], [0, 100, 0, 100]]).T).reshape((2, 2)), alpha=.7)

plt.show()