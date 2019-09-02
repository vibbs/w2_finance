import math
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from matplotlib import style
# import seaborn as seabornInstance 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from datetime import datetime, timedelta
import pandas_datareader.data as web

from pandas import Series, DataFrame

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

# IMP: from the file system
# dataset = pd.read_csv('./data_files/COKE.csv')

start = datetime(2000, 1, 1)
end = datetime(2019, 9, 1)

dataset = web.DataReader("AAPL", 'yahoo', start, end)
dataset.tail()

# close_px = dataset['Adj Close']
# mavg = close_px.rolling(window=100).mean()

# close_px.plot(label='Coke')
# mavg.plot(label='mavg')
# plt.legend()
# # plt.show()



# Prediction code for the linear regression. 
# Below Code is being copied from the Tutorial blogpost
dfreg = dataset.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (dataset['High'] - dataset['Low']) / dataset['Close'] * 100.0
dfreg['PCT_change'] = (dataset['Close'] - dataset['Open']) / dataset['Open'] * 100.0

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)



# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col]
# .shift(-forecast_out)


X = np.array(dfreg.drop(['label'], 1))


# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)


# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)


confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)

# Testing the different models with 80-20 rule for checking the confidence level of the system
print('The linear regression confidence is ', confidencereg)
print('The quadratic regression 2 confidence is ', confidencepoly2)
print('The quadratic regression 3 confidence is ', confidencepoly3)
print('The knn regression confidence is ', confidenceknn)


# this could be any prediction model.
forecast_set = clfreg.predict(X_lately)

print('--------------forecast_set-------------')
print(forecast_set)
dfreg['Forecast'] = np.nan


last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + timedelta(days=1)



for i in forecast_set:
    next_date = next_unix
    next_unix += timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg['Adj Close'].tail(500).plot()

print(dfreg)
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()