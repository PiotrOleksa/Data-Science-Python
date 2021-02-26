from binance.client import Client
from BINANCE_API import BINANCE_API_KEY, BINANCE_SECRET_KEY
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
 



#----------------- Client
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

#----------------- GRT
#Get info
numbers = []
prices_GRT = []
qty_GRT = [] 
quoteQty_GRT = []


#500 List
for i in range(500):
    numbers.append(i)

#Price
for i in range(500):
    prices_GRT.append(client.get_recent_trades(symbol='GRTUSDT')[i]['price'])
    
for i in range(500):
    prices_GRT[i]=float(prices_GRT[i])


#qty
for i in range(500):
    qty_GRT.append(client.get_recent_trades(symbol='GRTUSDT')[i]['qty'])
    
for i in range(500):
    qty_GRT[i]=float(qty_GRT[i])    


#quoteQty
for i in range(500):
    quoteQty_GRT.append(client.get_recent_trades(symbol='GRTUSDT')[i]['quoteQty'])
    
for i in range(500):
    quoteQty_GRT[i]=float(quoteQty_GRT[i])


#DataFrame
df_GRT = pd.DataFrame({'Price_GRT': prices_GRT,
                      'qty_GRT': qty_GRT,
                      'quoteQty_GRT':quoteQty_GRT})

#Plot
#Price
plt.plot(numbers, prices_GRT)
plt.title('GRT Price')
plt.yticks(np.arange(min(prices_GRT), max(prices_GRT), 0.01))
plt.show()

#qty
plt.plot(numbers, qty_GRT)
plt.title('GRT qty')
plt.show()

#quoteQty
plt.plot(numbers, quoteQty_GRT)
plt.title('GRT quoteQty')
plt.show()

#Print
print(df_GRT.describe())
print(df_GRT.corr())

#Linear Regression
#Arrays
GRT_price_arr = df_GRT['Price_GRT'].to_numpy().reshape(-1, 1)
arr_500 = np.arange(500).reshape(-1, 1)
arr_550 = np.arange(501, 550).reshape(-1, 1)

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(arr_500, GRT_price_arr, test_size = 0.2, random_state=222)

#Building a Model
reg = LinearRegression(fit_intercept=True)
model = reg.fit(X_train, y_train) 
prediction_test = model.predict(X_test)

print('Coefficients: \n', '{:.7f}'.format(reg.coef_[0][0]))
print('Mean squared error: %.4f'
      % mean_squared_error(y_test, prediction_test))
print('Coefficient of determination: %.4f'
      % r2_score(y_test, prediction_test))

plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, prediction_test, color='blue', linewidth=3)
plt.title("GRT Linear Regression Test")
plt.grid()
plt.show()
 
prediction = model.predict(arr_550)
plt.plot(arr_550, prediction, color='red', linewidth=3)
plt.title("GRT Price Prediciton")
plt.grid()
plt.show()
