# Python code extracted from walmart-sales-forecasting.md

# Code Block 1
import numpy as np      # To use np.arrays
import pandas as pd     # To use dataframes
from pandas.plotting import autocorrelation_plot as auto_corr

# To plot
import matplotlib.pyplot as plt     
import matplotlib as mpl
import seaborn as sns

#For date-time
import math
from datetime import datetime
from datetime import timedelta

# Another imports if needs
import itertools
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from statsmodels.tsa.seasonal import seasonal_decompose as season


from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA


import warnings
warnings.filterwarnings("ignore")

# Code Block 2
pd.options.display.max_columns=100 # to see columns

# Code Block 3
df_store = pd.read_csv('stores.csv') #store data

# Code Block 4
df_train = pd.read_csv('train.csv') # train set

# Code Block 5
df_features = pd.read_csv('features.csv') #external information

# Code Block 6
# merging 3 different sets
df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
df.head(5)

# Code Block 7
df.drop(['IsHoliday_y'], axis=1,inplace=True) # removing dublicated column

# Code Block 8
df.rename(columns={'IsHoliday_x':'IsHoliday'},inplace=True) # rename the column

# Code Block 9
df = df.loc[df['Weekly_Sales'] > 0]

# Code Block 10
df.shape # new data shape

# Code Block 11
df['Date'] = pd.to_datetime(df['Date']) # convert to datetime
#df['week'] =df['Date'].dt.week
df['month'] =df['Date'].dt.month 
df['year'] =df['Date'].dt.year

# Code Block 12
# Super bowl dates in train set
df.loc[(df['Date'] == '2010-02-12')|(df['Date'] == '2011-02-11')|(df['Date'] == '2012-02-10'),'Super_Bowl'] = True
df.loc[(df['Date'] != '2010-02-12')&(df['Date'] != '2011-02-11')&(df['Date'] != '2012-02-10'),'Super_Bowl'] = False

# Code Block 13
# Labor day dates in train set
df.loc[(df['Date'] == '2010-09-10')|(df['Date'] == '2011-09-09')|(df['Date'] == '2012-09-07'),'Labor_Day'] = True
df.loc[(df['Date'] != '2010-09-10')&(df['Date'] != '2011-09-09')&(df['Date'] != '2012-09-07'),'Labor_Day'] = False

# Code Block 14
# Thanksgiving dates in train set
df.loc[(df['Date'] == '2010-11-26')|(df['Date'] == '2011-11-25'),'Thanksgiving'] = True
df.loc[(df['Date'] != '2010-11-26')&(df['Date'] != '2011-11-25'),'Thanksgiving'] = False

# Code Block 15
#Christmas dates in train set
df.loc[(df['Date'] == '2010-12-31')|(df['Date'] == '2011-12-30'),'Christmas'] = True
df.loc[(df['Date'] != '2010-12-31')&(df['Date'] != '2011-12-30'),'Christmas'] = False

# Code Block 16
df = df.fillna(0) # filling null's with 0

# Code Block 17
df.to_csv('clean_data.csv') # assign new data frame to csv for using after here

# Code Block 18
df["Date"] = pd.to_datetime(df["Date"]) #changing data to datetime for decomposing

# Code Block 19
df.set_index('Date', inplace=True) #seting date as index

# Code Block 20
df_week = df.resample('W').mean() #resample data as weekly

# Code Block 21
df_week_diff = df_week['Weekly_Sales'].diff().dropna() #creating difference values

# Code Block 22
train_data_diff = df_week_diff [:int(0.7*(len(df_week_diff )))]
test_data_diff = df_week_diff [int(0.7*(len(df_week_diff ))):]

# Code Block 23
model_holt_winters = ExponentialSmoothing(train_data_diff, seasonal_periods=20, seasonal='additive',
                                           trend='additive',damped=True).fit() #Taking additive trend and seasonality.
y_pred = model_holt_winters.forecast(len(test_data_diff))# Predict the test data

# Code Block 24
def wmae_test(test, pred): # WMAE for test 
    error = np.mean(np.abs(test - pred), axis=0)
    return error

# Code Block 25
wmae_test(test_data_diff, y_pred)

# End of extracted code
