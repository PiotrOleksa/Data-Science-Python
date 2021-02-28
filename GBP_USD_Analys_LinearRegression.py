import pandas as pd
from pandas.plotting import table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore


#---------------------- Import the data
eur_gbp = pd.read_csv('eurgbp.csv', sep=',')
usd_pln = pd.read_csv('usdpln.csv', sep=',')




#---------------------- Variables
max_eur_gbp = round(eur_gbp['Close'].max(), 5)
mean_eur_gbp = round(eur_gbp['Close'].mean(), 5)
min_eur_gbp = round(eur_gbp['Close'].min(), 5)

max_usd_pln = round(usd_pln['Close'].max(), 5)
mean_usd_pln = round(usd_pln['Close'].mean(), 5)
min_usd_pln = round(usd_pln['Close'].min(), 5)

eur_gbp_nodate = eur_gbp.drop(columns=['Date'])
usd_pln_nodate = usd_pln.drop(columns=['Date'])

common_close_price = pd.DataFrame({'EUR_GBP_CLOSE':eur_gbp['Close'],
                                   'USD_PLN_CLOSE':usd_pln['Close']})


#---------------------- Exploring
#Describe 
eur_gbp_inf = eur_gbp.describe()
usd_pln_inf = usd_pln.describe()

sub_plot = plt.subplot(111, frame_on=False)

sub_plot.xaxis.set_visible(False) 
sub_plot.yaxis.set_visible(False) 



table(sub_plot, eur_gbp_inf, loc='upper right').scale(1, 2)
plt.title("EUR-GRB Describe")
plt.savefig('eur_gbp_inf.png', dpi=199)

table(sub_plot, usd_pln_inf, loc='upper right').scale(1, 2)
plt.title("USD-PLN Describe")
plt.savefig('usd_pln_inf.png', dpi=199)


#Prince in time plot
#EUR_GBP
xdates = [dt.strptime(dstr,'%Y-%m-%d') for dstr in eur_gbp['Date']]
plt.plot(xdates, eur_gbp['Close'])
plt.axhline(y=max_eur_gbp,label=f'Higherst Price: {max_eur_gbp}' ,color='r', linestyle='--')
plt.axhline(y=mean_eur_gbp,label=f'Mean Price: {mean_eur_gbp}' ,color='g', linestyle='--')
plt.axhline(y=min_eur_gbp,label=f'Lowest Price: {min_eur_gbp}' ,color='y', linestyle='--')
plt.legend()
plt.title("EUR Close Price in GBP")
plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.grid()
plt.savefig('eur_gpb_plot.png', dpi=199)

#USD_PLN
xdates = [dt.strptime(dstr,'%Y-%m-%d') for dstr in usd_pln['Date']]
plt.plot(xdates, usd_pln['Close'])
plt.axhline(y=max_usd_pln,label=f'Higherst Price: {max_usd_pln}' ,color='r', linestyle='--')
plt.axhline(y=mean_usd_pln,label=f'Mean Price: {mean_usd_pln}' ,color='g', linestyle='--')
plt.axhline(y=min_usd_pln,label=f'Lowest Price: {min_usd_pln}' ,color='y', linestyle='--')
plt.legend()
plt.title("USD Close Price in PLN")
plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.grid()
plt.savefig('usd_pln_plot.png', dpi=199)


#EUR_GBP to USD_PLN scatter
plt.scatter(eur_gbp['Close'], usd_pln['Close'])
plt.title("EUR-GRB to USD-PLN")
plt.savefig('EUR-GRB to USD-PLN.png', dpi=199)

#Removing outliners data
z_scores_eur = zscore(eur_gbp['Close'])
abs_z_scores = np.abs(z_scores_eur)
filtered_entries = (abs_z_scores.reshape(259, 1) < 3).all(axis= 1)
eur_gbp_out = eur_gbp['Close'][filtered_entries]


z_scores_usd = zscore(usd_pln['Close'])
abs_z_scores = np.abs(z_scores_eur)
filtered_entries = (abs_z_scores.reshape(259, 1) < 3).all(axis= 1)
usd_pln_out = usd_pln['Close'][filtered_entries]

z_scores_common = zscore(common_close_price)
abs_z_scores = np.abs(z_scores_common)
filtered_entries = (abs_z_scores < 2).all(axis= 1)
common_close_price_out = common_close_price[filtered_entries]

plt.scatter(common_close_price_out['EUR_GBP_CLOSE'], common_close_price_out['USD_PLN_CLOSE'])
plt.title("EUR-GRB to USD-PLN")
plt.savefig('EUR-GRB to USD-PLN.png', dpi=199)

#Correlation
common_close_price_out.corr()


plt.scatter(common_close_price_out['EUR_GBP_CLOSE'][-50:], common_close_price_out['USD_PLN_CLOSE'][-50:])
plt.title("EUR-GRB to USD-PLN last 50 days")
plt.savefig('EUR-GRB to USD-PLN last 50 days.png', dpi=199)


#Creating a model
eur_gbp_array = eur_gbp['Close'].to_numpy().reshape(-1, 1)
usd_pln_array = usd_pln['Close'].to_numpy().reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(eur_gbp_array, usd_pln_array, test_size = 0.2, random_state=222)

reg = LinearRegression(fit_intercept=True)
model = reg.fit(X_train, y_train) 
prediction = model.predict(X_test)

print('Coefficients: \n', reg.coef_)
print('Mean squared error: %.4f'
      % mean_squared_error(y_test, prediction))
print('Coefficient of determination: %.4f'
      % r2_score(y_test, prediction))


plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, prediction, color='blue', linewidth=3)
plt.title("Linear Regression EUR-GBP to USD-PLN")
plt.grid()
plt.savefig('EUR-GRB to USD-PLN Linear Regression.png', dpi=199)

model.predict([[0.88]])

























