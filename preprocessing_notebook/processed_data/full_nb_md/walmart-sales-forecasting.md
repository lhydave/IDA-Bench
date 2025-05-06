# Background

Walmart is a renowned retail corporation that operates a chain of hypermarkets. Here, Walmart has provided a data combining of 45 stores including store information and monthly sales. The data is provided on weekly basis. Walmart tries to find the impact of holidays on the sales of store. For which it has included four holidays’ weeks into the dataset which are Christmas, Thanksgiving, Super bowl, Labor Day. Here we are owing to Analyze the dataset given. Before doing that, let me point out the objective of this analysis. 

# Business Objectives

Our Main Objective is to predict sales of store in a week. As in dataset size and time related data are given as feature, so analyze if sales are impacted by time-based factors and space- based factor. Most importantly how inclusion of holidays in a week soars the sales in store? 

# Importing Necessary Libraries and Data


```python
import numpy as np      # To use np.arrays
import pandas as pd     # To use dataframes
from pandas.plotting import autocorrelation_plot as auto_corr

# To plot
import matplotlib.pyplot as plt  
%matplotlib inline    
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

from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose as season
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
!pip install pmdarima
from pmdarima.utils import decomposed_plot
from pmdarima.arima import decompose
from pmdarima import auto_arima


import warnings
warnings.filterwarnings("ignore")


```

    [33mWARNING: Ignoring invalid distribution ~ympy (/home/tianyu/miniconda3/lib/python3.12/site-packages)[0m[33m
    [0m[33mWARNING: Ignoring invalid distribution ~ympy (/home/tianyu/miniconda3/lib/python3.12/site-packages)[0m[33m
    [0mRequirement already satisfied: pmdarima in /home/tianyu/miniconda3/lib/python3.12/site-packages (2.0.4)
    Requirement already satisfied: joblib>=0.11 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pmdarima) (1.4.2)
    Requirement already satisfied: Cython!=0.29.18,!=0.29.31,>=0.29 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pmdarima) (3.0.12)
    Requirement already satisfied: numpy>=1.21.2 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pmdarima) (2.2.2)
    Requirement already satisfied: pandas>=0.19 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pmdarima) (2.2.3)
    Requirement already satisfied: scikit-learn>=0.22 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pmdarima) (1.6.1)
    Requirement already satisfied: scipy>=1.3.2 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pmdarima) (1.15.1)
    Requirement already satisfied: statsmodels>=0.13.2 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pmdarima) (0.14.4)
    Requirement already satisfied: urllib3 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pmdarima) (2.2.3)
    Requirement already satisfied: setuptools!=50.0.0,>=38.6.0 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pmdarima) (72.1.0)
    Requirement already satisfied: packaging>=17.1 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pmdarima) (24.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pandas>=0.19->pmdarima) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pandas>=0.19->pmdarima) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from pandas>=0.19->pmdarima) (2023.3)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from scikit-learn>=0.22->pmdarima) (3.6.0)
    Requirement already satisfied: patsy>=0.5.6 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from statsmodels>=0.13.2->pmdarima) (1.0.1)
    Requirement already satisfied: six>=1.5 in /home/tianyu/miniconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas>=0.19->pmdarima) (1.16.0)
    [33mWARNING: Ignoring invalid distribution ~ympy (/home/tianyu/miniconda3/lib/python3.12/site-packages)[0m[33m
    [0m[33mWARNING: Ignoring invalid distribution ~ympy (/home/tianyu/miniconda3/lib/python3.12/site-packages)[0m[33m
    [0m[33mWARNING: Ignoring invalid distribution ~ympy (/home/tianyu/miniconda3/lib/python3.12/site-packages)[0m[33m
    [0m[33mWARNING: Ignoring invalid distribution ~ympy (/home/tianyu/miniconda3/lib/python3.12/site-packages)[0m[33m
    [0m


```python
pd.options.display.max_columns=100 # to see columns 
```


```python
df_store = pd.read_csv('stores.csv') #store data
```


```python
df_train = pd.read_csv('train.csv') # train set
```


```python
df_features = pd.read_csv('features.csv') #external information
```

# First Look to Data and Merging Three Dataframes


```python
df_store.head()
```









```python
df_train.head()
```









```python
df_features.head()
```









```python
# merging 3 different sets
df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
df.head(5)
```









```python
df.drop(['IsHoliday_y'], axis=1,inplace=True) # removing dublicated column
```


```python
df.rename(columns={'IsHoliday_x':'IsHoliday'},inplace=True) # rename the column
```


```python
df.head() # last ready data set
```









```python
df.shape
```




    (421570, 16)



# Store & Department Numbers


```python
df['Store'].nunique() # number of different values
```




    45




```python
df['Dept'].nunique() # number of different values
```




    81



Now, I will look at the average weekly sales for each store and each department to see if there is any weird values or not. There are 45 stores and 81 departments for stores. 


```python
store_dept_table = pd.pivot_table(df, index='Store', columns='Dept',
                                  values='Weekly_Sales', aggfunc=np.mean)
display(store_dept_table)
```





Store numbers begin from 1 to 45, department numbers are from 1 to 99, but some numbers are missing such as there is no 88 or 89 etc. Total number of departments is 81. 

From the pivot table, it is obviously seen that there are some wrong values such as there are 0 and minus values for weekly sales. But sales amount can not be minus. Also, it is impossible for one department not to sell anything whole week. So, I will change this values.


```python
df.loc[df['Weekly_Sales']<=0]
```








1358 rows in 421570 rows means 0.3%, so I can delete and ignore these rows which contains wrong sales values.


```python
df = df.loc[df['Weekly_Sales'] > 0]
```


```python
df.shape # new data shape
```




    (420212, 16)



# Date


```python
df['Date'].head(5).append(df['Date'].tail(5)) # to see first and last 5 rows.
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_1224063/4262877864.py in ?()
    ----> 1 df['Date'].head(5).append(df['Date'].tail(5)) # to see first and last 5 rows.


    ~/DataSciBench/.venv/lib/python3.12/site-packages/pandas/core/generic.py in ?(self, name)
       6295             and name not in self._accessors
       6296             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6297         ):
       6298             return self[name]
    -> 6299         return object.__getattribute__(self, name)


    AttributeError: 'Series' object has no attribute 'append'


Our data is from 5th of February 2010 to 26th of October 2012.  

# IsHoliday column


```python
sns.barplot(x='IsHoliday', y='Weekly_Sales', data=df)
```




    <Axes: xlabel='IsHoliday', ylabel='Weekly_Sales'>




    
    



```python
df_holiday = df.loc[df['IsHoliday']==True]
df_holiday['Date'].unique() 
```




    array(['2010-02-12', '2010-09-10', '2010-11-26', '2010-12-31',
           '2011-02-11', '2011-09-09', '2011-11-25', '2011-12-30',
           '2012-02-10', '2012-09-07'], dtype=object)




```python
df_not_holiday = df.loc[df['IsHoliday']==False]
df_not_holiday['Date'].nunique() 
```




    133



All holidays are not in the data. There are 4 holiday values such as;

Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13

Labor Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13

Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13

Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13


After the 07-Sep-2012 holidays are in test set for prediction. When we look at the data, average weekly sales for holidays are significantly higher than not-holiday days. In train data, there are 133 weeks for non-holiday and 10 weeks for holiday.

I want to see differences between holiday types. So, I create new columns for 4 types of holidays and fill them with boolean values. If date belongs to this type of holiday it is True, if not False. 


```python
# Super bowl dates in train set
df.loc[(df['Date'] == '2010-02-12')|(df['Date'] == '2011-02-11')|(df['Date'] == '2012-02-10'),'Super_Bowl'] = True
df.loc[(df['Date'] != '2010-02-12')&(df['Date'] != '2011-02-11')&(df['Date'] != '2012-02-10'),'Super_Bowl'] = False
```


```python
# Labor day dates in train set
df.loc[(df['Date'] == '2010-09-10')|(df['Date'] == '2011-09-09')|(df['Date'] == '2012-09-07'),'Labor_Day'] = True
df.loc[(df['Date'] != '2010-09-10')&(df['Date'] != '2011-09-09')&(df['Date'] != '2012-09-07'),'Labor_Day'] = False
```


```python
# Thanksgiving dates in train set
df.loc[(df['Date'] == '2010-11-26')|(df['Date'] == '2011-11-25'),'Thanksgiving'] = True
df.loc[(df['Date'] != '2010-11-26')&(df['Date'] != '2011-11-25'),'Thanksgiving'] = False
```


```python
#Christmas dates in train set
df.loc[(df['Date'] == '2010-12-31')|(df['Date'] == '2011-12-30'),'Christmas'] = True
df.loc[(df['Date'] != '2010-12-31')&(df['Date'] != '2011-12-30'),'Christmas'] = False
```


```python
sns.barplot(x='Christmas', y='Weekly_Sales', data=df) # Christmas holiday vs not-Christmas
```




    <Axes: xlabel='Christmas', ylabel='Weekly_Sales'>




    
    



```python
sns.barplot(x='Thanksgiving', y='Weekly_Sales', data=df) # Thanksgiving holiday vs not-thanksgiving
```




    <Axes: xlabel='Thanksgiving', ylabel='Weekly_Sales'>




    
    



```python
sns.barplot(x='Super_Bowl', y='Weekly_Sales', data=df) # Super bowl holiday vs not-super bowl
```




    <Axes: xlabel='Super_Bowl', ylabel='Weekly_Sales'>




    
    



```python
sns.barplot(x='Labor_Day', y='Weekly_Sales', data=df) # Labor day holiday vs not-labor day
```




    <Axes: xlabel='Labor_Day', ylabel='Weekly_Sales'>




    
    


It is shown that for the graphs, Labor Day and Christmas do not increase weekly average sales. There is positive effect on sales in Super bowl, but the highest difference is in the Thanksgiving. I think, people generally prefer to buy Christmas gifts 1-2 weeks before Christmas, so it does not change sales in the Christmas week. And, there is Black Friday sales in the Thanksgiving week.

# Type Effect on Holidays

There are three different store types in the data as A, B and C.


```python
df.groupby(['Christmas','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Christmas 
```




    Christmas  Type
    False      A       20174.350209
               B       12301.986116
               C        9570.951973
    True       A       18310.167535
               B       11488.988057
               C        8031.520607
    Name: Weekly_Sales, dtype: float64




```python
df.groupby(['Labor_Day','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Labor Day
```




    Labor_Day  Type
    False      A       20151.210941
               B       12294.954138
               C        9542.098293
    True       A       20004.267422
               B       12084.304642
               C        9893.459258
    Name: Weekly_Sales, dtype: float64




```python
df.groupby(['Thanksgiving','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Thanksgiving
```




    Thanksgiving  Type
    False         A       20044.007801
                  B       12197.717405
                  C        9547.377807
    True          A       27397.776346
                  B       18733.973971
                  C        9696.566616
    Name: Weekly_Sales, dtype: float64




```python
df.groupby(['Super_Bowl','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Super Bowl
```




    Super_Bowl  Type
    False       A       20138.055908
                B       12286.739293
                C        9536.110508
    True        A       20612.757674
                B       12467.035506
                C       10179.271884
    Name: Weekly_Sales, dtype: float64



I want to see percentages of store types.


```python
my_data = [48.88, 37.77 , 13.33 ]  #percentages
my_labels = 'Type A','Type B', 'Type C' # labels
plt.pie(my_data,labels=my_labels,autopct='%1.1f%%', textprops={'fontsize': 15}) #plot pie type and bigger the labels
plt.axis('equal')
mpl.rcParams.update({'font.size': 20}) #bigger percentage labels

plt.show()
```


    
    



```python
df.groupby('IsHoliday')['Weekly_Sales'].mean()
```




    IsHoliday
    False    15952.816352
    True     17094.300918
    Name: Weekly_Sales, dtype: float64



Nearly, half of the stores are belongs to Type A.


```python
# Plotting avg wekkly sales according to holidays by types
plt.style.use('seaborn-poster')
labels = ['Thanksgiving', 'Super_Bowl', 'Labor_Day', 'Christmas']
A_means = [27397.77, 20612.75, 20004.26, 18310.16]
B_means = [18733.97, 12463.41, 12080.75, 11483.97]
C_means = [9696.56,10179.27,9893.45,8031.52]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(16, 8))
rects1 = ax.bar(x - width, A_means, width, label='Type_A')
rects2 = ax.bar(x , B_means, width, label='Type_B')
rects3 = ax.bar(x + width, C_means, width, label='Type_C')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Weekly Avg Sales')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.axhline(y=17094.30,color='r') # holidays avg
plt.axhline(y=15952.82,color='green') # not-holiday avg

fig.tight_layout()

plt.show()
```


    
    


It is seen from the graph that, highest sale average is in the Thanksgiving week between holidays. And, for all holidays Type A stores has highest sales.


```python
df.sort_values(by='Weekly_Sales',ascending=False).head(5)
```








Also, it is not surprise that top 5 highest weekly sales are belongs to Thanksgiving weeks.

# To See the Size - Type Relation


```python
df_store.groupby('Type').describe()['Size'].round(2) # See the Size-Type relation
```









```python
plt.figure(figsize=(10,8)) # To see the type-size relation
fig = sns.boxplot(x='Type', y='Size', data=df, showfliers=False)
```


    
    


Size of the type of stores are consistent with sales, as expected. Higher size stores has higher sales. And, Walmart classify stores according to their sizes according to graph. After the smallest size value of Type A, Type B begins. After the smallest size value of Type B, Type C begins.

# Markdown Columns

Walmart gave markdown columns to see the effect if markdowns on sales. When I check columns, there are many NaN values for markdowns. I decided to change them with 0, because if there is markdown in the row, it is shown with numbres. So, if I can write 0, it shows there is no markdown at that date.


```python
df.isna().sum()
```




    Store                0
    Dept                 0
    Date                 0
    Weekly_Sales         0
    IsHoliday            0
    Temperature          0
    Fuel_Price           0
    MarkDown1       270031
    MarkDown2       309308
    MarkDown3       283561
    MarkDown4       285694
    MarkDown5       269283
    CPI                  0
    Unemployment         0
    Type                 0
    Size                 0
    Super_Bowl           0
    Labor_Day            0
    Thanksgiving         0
    Christmas            0
    dtype: int64




```python
df = df.fillna(0) # filling null's with 0
```


```python
df.isna().sum() # last null check
```




    Store           0
    Dept            0
    Date            0
    Weekly_Sales    0
    IsHoliday       0
    Temperature     0
    Fuel_Price      0
    MarkDown1       0
    MarkDown2       0
    MarkDown3       0
    MarkDown4       0
    MarkDown5       0
    CPI             0
    Unemployment    0
    Type            0
    Size            0
    Super_Bowl      0
    Labor_Day       0
    Thanksgiving    0
    Christmas       0
    dtype: int64




```python
df.describe() # to see weird statistical things
```








Minimum value for weekly sales is 0.01. Most probably, this value is not true but I prefer not to change them now. Because, there are many departments and many stores. It takes too much time to check each department for each store (45 store for 81 departments). So, I take averages for EDA. 

# Deeper Look in Sales


```python
x = df['Dept']
y = df['Weekly_Sales']
plt.figure(figsize=(15,5))
plt.title('Weekly Sales by Department')
plt.xlabel('Departments')
plt.ylabel('Weekly Sales')
plt.scatter(x,y)
plt.show()
```


    
    



```python
plt.figure(figsize=(30,10))
fig = sns.barplot(x='Dept', y='Weekly_Sales', data=df)
```


    
    


From the first graph, it is seen that one department between 60-80(I assume it is 72), has higher sales values. But, when we take the averages, it is seen that department 92 has higher mean sales. Department 72 is seasonal department, I think. It has higher values is some seasons but on average 92 is higher.


```python
x = df['Store']
y = df['Weekly_Sales']
plt.figure(figsize=(15,5))
plt.title('Weekly Sales by Store')
plt.xlabel('Stores')
plt.ylabel('Weekly Sales')
plt.scatter(x,y)
plt.show()
```


    
    



```python
plt.figure(figsize=(20,6))
fig = sns.barplot(x='Store', y='Weekly_Sales', data=df)
```


    
    


Same thing happens in stores. From the first graph, some stores has higher sales but on average store 20 is the best and 4 and 14 following it.

# Changing Date to Datetime and Creating New Columns


```python
df["Date"] = pd.to_datetime(df["Date"]) # convert to datetime
df['week'] =df['Date'].dt.week
df['month'] =df['Date'].dt.month 
df['year'] =df['Date'].dt.year
```


```python
df.groupby('month')['Weekly_Sales'].mean() # to see the best months for sales
```




    month
    1     14182.239153
    2     16048.701191
    3     15464.817698
    4     15696.435193
    5     15845.556200
    6     16397.605478
    7     15905.472425
    8     16113.800069
    9     15147.216063
    10    15279.182119
    11    17534.964277
    12    19425.798603
    Name: Weekly_Sales, dtype: float64




```python
df.groupby('year')['Weekly_Sales'].mean() # to see the best years for sales
```




    year
    2010    16318.648285
    2011    16007.797985
    2012    15748.265005
    Name: Weekly_Sales, dtype: float64




```python
monthly_sales = pd.pivot_table(df, values = "Weekly_Sales", columns = "year", index = "month")
monthly_sales.plot()
```




    <Axes: xlabel='month'>




    
    


From the graph, it is seen that 2011 has lower sales than 2010 generally. When we look at the mean sales it is seen that 2010 has higher values, but 2012 has no information about November and December which have higher sales. Despite of 2012 has no last two months sales, it's mean is near to 2010. Most probably, it will take the first place if we get 2012 results and add them.


```python
fig = sns.barplot(x='month', y='Weekly_Sales', data=df)
```


    
    


When we look at the graph above, the best sales are in December and November, as expected. The highest values are belongs to Thankgiving holiday but when we take average it is obvious that December has the best value.


```python
df.groupby('week')['Weekly_Sales'].mean().sort_values(ascending=False).head()
```




    week
    51    26454.164116
    47    22269.601768
    50    20478.421134
    49    18731.794840
    22    16856.650245
    Name: Weekly_Sales, dtype: float64



Top 5 sales averages by weekly belongs to 1-2 weeks before Christmas, Thanksgiving, Black Friday and end of May, when the schools are closed. 


```python
weekly_sales = pd.pivot_table(df, values = "Weekly_Sales", columns = "year", index = "week")
weekly_sales.plot()
```




    <Axes: xlabel='week'>




    
    



```python
plt.figure(figsize=(20,6))
fig = sns.barplot(x='week', y='Weekly_Sales', data=df)
```


    
    


From graphs, it is seen that 51th week and 47th weeks have significantly higher averages as Christmas, Thankgiving and Black Friday effects.

# Fuel Price, CPI , Unemployment , Temperature Effects


```python
fuel_price = pd.pivot_table(df, values = "Weekly_Sales", index= "Fuel_Price")
fuel_price.plot()
```




    <Axes: xlabel='Fuel_Price'>




    
    



```python
temp = pd.pivot_table(df, values = "Weekly_Sales", index= "Temperature")
temp.plot()
```




    <Axes: xlabel='Temperature'>




    
    



```python
CPI = pd.pivot_table(df, values = "Weekly_Sales", index= "CPI")
CPI.plot()
```




    <Axes: xlabel='CPI'>




    
    



```python
unemployment = pd.pivot_table(df, values = "Weekly_Sales", index= "Unemployment")
unemployment.plot()
```




    <Axes: xlabel='Unemployment'>




    
    


From graphs, it is seen that there are no significant patterns between CPI, temperature, unemployment rate, fuel price vs weekly sales. There is no data for CPI between 140-180 also.


```python
df.to_csv('clean_data.csv') # assign new data frame to csv for using after here
```

# Findings and Explorations

# Cleaning Process

- The data has no too much missing values. All columns was checked. 
- I choose rows which has higher than 0 weekly sales. Minus values are 0.3% of data. So, I dropped them.
- Null values in markdowns changed to zero. Because, they were written as null if there were no markdown on this department. 

# Explorations & Findings

- There are 45 stores and 81 department in data. Departments are not same in all stores. 
- Although department 72 has higher weekly sales values, on average department 92 is the best. It shows us, some departments has higher values as seasonal like Thanksgiving. It is consistant when we look at the top 5 sales in data, all of them belongs to 72th department at Thanksgiving holiday time. 
- Although stores 10 and 35 have higher weekly sales values sometimes, in general average store 20 and store 4 are on the first and second rank. It means that some areas has higher seasonal sales. 
- Stores has 3 types as A, B and C according to their sizes. Almost half of the stores are bigger than 150000 and categorized as A. According to type, sales of the stores are changing.
- As expected, holiday average sales are higher than normal dates.
- Christmas holiday introduces as the last days of the year. But people generally shop at 51th week. So, when we look at the total sales of holidays, Thankgiving has higher sales between them which was assigned by Walmart.
- Year 2010 has higher sales than 2011 and 2012. But, November and December sales are not in the data for 2012. Even without highest sale months, 2012 is not significantly less than 2010, so after adding last two months, it can be first.
- It is obviously seen that week 51 and 47 have higher values and 50-48 weeks follow them. Interestingly, 5th top sales belongs to 22th week of the year. This results show that Christmas, Thankgiving and Black Friday are very important than other weeks for sales and 5th important time is 22th week of the year and it is end of the May, when schools are closed. Most probably, people are preparing for holiday at the end of the May. 
- January sales are significantly less than other months. This is the result of November and December high sales. After two high sales month, people prefer to pay less on January.
- CPI, temperature, unemployment rate and fuel price have no pattern on weekly sales. 


# First Trial with Random Forest

Generally, Rondom Forest Regressor gives good results when we tune it well. So, to find simple baseline model, I will use RandomForestRegressor in this notebook. Also, feature importance for model can be found in this notebook. 

Our metric for this project is weighted mean absolute error (WMAE):

![title](https://miro.medium.com/max/990/1*VKYKK85ViLYUUjyOWVURfw.jpeg)

where

- n is the number of rows
- ŷ i is the predicted sales
- yi is the actual sales
- wi are weights. w = 5 if the week is a holiday week, 1 otherwise

With this metric, the error at holiday weeks has 5 times weight more than normal weeks. So, it is more important to predict sales at holiday weeks accurately.
All results for trails can be found at the end of this notebook.


```python
pd.options.display.max_columns=100 # to see columns 
```


```python
df = pd.read_csv('./clean_data.csv')
```


```python
df.drop(columns=['Unnamed: 0'],inplace=True)
```


```python
df['Date'] = pd.to_datetime(df['Date']) # changing datetime to divide if needs
```

# Encoding the Data 

For preprocessing our data, I will change holidays boolean values to 0-1 and replace type of the stores from A, B, C to 1, 2, 3. 


```python
df_encoded = df.copy() # to keep original dataframe taking copy of it
```


```python
type_group = {'A':1, 'B': 2, 'C': 3}  # changing A,B,C to 1-2-3
df_encoded['Type'] = df_encoded['Type'].replace(type_group)
```


```python
df_encoded['Super_Bowl'] = df_encoded['Super_Bowl'].astype(bool).astype(int) # changing T,F to 0-1
```


```python
df_encoded['Thanksgiving'] = df_encoded['Thanksgiving'].astype(bool).astype(int) # changing T,F to 0-1
```


```python
df_encoded['Labor_Day'] = df_encoded['Labor_Day'].astype(bool).astype(int) # changing T,F to 0-1
```


```python
df_encoded['Christmas'] = df_encoded['Christmas'].astype(bool).astype(int) # changing T,F to 0-1
```


```python
df_encoded['IsHoliday'] = df_encoded['IsHoliday'].astype(bool).astype(int) # changing T,F to 0-1
```


```python
df_new = df_encoded.copy() # taking the copy of encoded df to keep it original
```

# Observation of Interactions between Features

Firstly, i will drop divided holiday columns from my data and try without them. To keep my encoded data safe, I assigned my dataframe to new one and I will use for this. 


```python
drop_col = ['Super_Bowl','Labor_Day','Thanksgiving','Christmas']
df_new.drop(drop_col, axis=1, inplace=True) # dropping columns
```


```python
plt.figure(figsize = (12,10))
sns.heatmap(df_new.corr().abs())    # To see the correlations
plt.show()
```


    
    


Temperature, unemployment, CPI have no significant effect on weekly sales, so I will drop them. Also, Markdown 4 and 5 highly correlated with Markdown 1. So, I will drop them also. It can create multicollinearity problem, maybe. So, first I will try without them.


```python
drop_col = ['Temperature','MarkDown4','MarkDown5','CPI','Unemployment']
df_new.drop(drop_col, axis=1, inplace=True) # dropping columns
```


```python
plt.figure(figsize = (12,10))
sns.heatmap(df_new.corr().abs())    # To see the correlations without dropping columns
plt.show()
```


    
    


Size and type are highly correlated with weekly sales. Also, department and store are correlated with sales.


```python
df_new = df_new.sort_values(by='Date', ascending=True) # sorting according to date
```

# Creating Train-Test Splits

Our date column has continuos values, to keep the date features continue, I will not take random splitting. so, I split data manually according to 70%.


```python
train_data = df_new[:int(0.7*(len(df_new)))] # taking train part
test_data = df_new[int(0.7*(len(df_new))):] # taking test part

target = "Weekly_Sales"
used_cols = [c for c in df_new.columns.to_list() if c not in [target]] # all columns except weekly sales

X_train = train_data[used_cols]
X_test = test_data[used_cols]
y_train = train_data[target]
y_test = test_data[target]
```


```python
X = df_new[used_cols] # to keep train and test X values together
```

We have enough information in our date such as week of the year. So, I drop date columns.


```python
X_train = X_train.drop(['Date'], axis=1) # dropping date from train
X_test = X_test.drop(['Date'], axis=1) # dropping date from test
```

# Metric Definition Function

Our metric is not calculated as default from ready models. It is weighed error so, I will use function below to calculate it.


```python
def wmae_test(test, pred): # WMAE for test 
    weights = X_test['IsHoliday'].apply(lambda is_holiday:5 if is_holiday else 1)
    error = np.sum(weights * np.abs(test - pred), axis=0) / np.sum(weights)
    return error
```

# Random Forest Regressor

To tune the regressor, I can use gridsearch but it takes too much time for this type of data which has many rows and columns. So, I choose regressor parameters manually. I changed the parameters each time and try to find the best result.


```python
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                           max_features = 'sqrt',min_samples_split = 10)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()



#making pipe tp use scaler and regressor together
pipe = make_pipeline(scaler,rf)

pipe.fit(X_train, y_train)

# predictions on train set
y_pred = pipe.predict(X_train)

# predictions on test set
y_pred_test = pipe.predict(X_test)
```


```python
wmae_test(y_test, y_pred_test)
```




    5850.444413125214



For the first trial, my weighted error is around 5850.

# To See Feature Importance


```python
X = X.drop(['Date'], axis=1) #dropping date column from X
```

Below code cell was taken from our instructor Bryan Arnold's notebook. I changed the code according to my data and see the plot.


```python
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Printing the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plotting the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
```

    Feature ranking:
    1. feature 1 (0.732822)
    2. feature 8 (0.110390)
    3. feature 0 (0.054027)
    4. feature 7 (0.038210)
    5. feature 9 (0.021277)
    6. feature 3 (0.018402)
    7. feature 10 (0.009446)
    8. feature 6 (0.005523)
    9. feature 4 (0.003413)
    10. feature 5 (0.002776)
    11. feature 2 (0.002246)
    12. feature 11 (0.001467)



    
    


After looking feature importance, I dropped least important 3-4 features and tried the model. I found the best result when I dropped month column which is highly correlated with week.


```python
X1_train = X_train.drop(['month'], axis=1) # dropping month
X1_test = X_test.drop(['month'], axis=1)
```

# Model Again without Month


```python
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                           max_features = 'sqrt',min_samples_split = 10)

scaler=RobustScaler()
pipe = make_pipeline(scaler,rf)

pipe.fit(X1_train, y_train)

# predictions on train set
y_pred = pipe.predict(X1_train)

# predictions on test set
y_pred_test = pipe.predict(X1_test)
```


```python
wmae_test(y_test, y_pred_test)
```




    5494.419090545123



It gives better results than baseline.

# Model with Whole Data

Now, I want to make sure that my model will learn from the columns which I dropped or not. So, I will apply my model to whole encoded data again.


```python
# splitting train-test to whole dataset
train_data_enc = df_encoded[:int(0.7*(len(df_encoded)))]
test_data_enc = df_encoded[int(0.7*(len(df_encoded))):]

target = "Weekly_Sales"
used_cols1 = [c for c in df_encoded.columns.to_list() if c not in [target]] # all columns except price

X_train_enc = train_data_enc[used_cols1]
X_test_enc = test_data_enc[used_cols1]
y_train_enc = train_data_enc[target]
y_test_enc = test_data_enc[target]
```


```python
X_enc = df_encoded[used_cols1] # to get together train,test splits
```


```python
X_enc = X_enc.drop(['Date'], axis=1) #dropping date column for whole X
```


```python
X_train_enc = X_train_enc.drop(['Date'], axis=1) # dropping date from train and test
X_test_enc= X_test_enc.drop(['Date'], axis=1)
```


```python
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                           max_features = 'sqrt',min_samples_split = 10)

scaler=RobustScaler()
pipe = make_pipeline(scaler,rf)

pipe.fit(X_train_enc, y_train_enc)

# predictions on train set
y_pred_enc = pipe.predict(X_train_enc)

# predictions on test set
y_pred_test_enc = pipe.predict(X_test_enc)
```


```python
wmae_test(y_test_enc, y_pred_test_enc)
```




    2450.1012493925496



We found better results for whole data, it means our model can learn from columns which I dropped before.

# Feature Importance for Whole Encoded Dataset


```python
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Printing the feature ranking
print("Feature ranking:")

for f in range(X_enc.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plotting the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_enc.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_enc.shape[1]), indices)
plt.xlim([-1, X_enc.shape[1]])
plt.show()
```

    Feature ranking:
    1. feature 1 (0.743136)
    2. feature 13 (0.075801)
    3. feature 12 (0.043810)
    4. feature 0 (0.031172)
    5. feature 10 (0.028007)
    6. feature 11 (0.017039)
    7. feature 18 (0.012315)
    8. feature 3 (0.011079)
    9. feature 4 (0.009879)
    10. feature 19 (0.005534)
    11. feature 7 (0.004481)
    12. feature 9 (0.003979)
    13. feature 5 (0.003461)
    14. feature 8 (0.003315)
    15. feature 6 (0.002381)
    16. feature 16 (0.001363)
    17. feature 20 (0.001200)
    18. feature 2 (0.000879)
    19. feature 17 (0.000597)
    20. feature 14 (0.000290)
    21. feature 15 (0.000283)



    
    


According to feature importance, I dropped some columns from whole set and try my model again.


```python
df_encoded_new = df_encoded.copy() # taking copy of encoded data to keep it without change.
df_encoded_new.drop(drop_col, axis=1, inplace=True)
```

# Model According to Feature Importance


```python
#train-test splitting
train_data_enc_new = df_encoded_new[:int(0.7*(len(df_encoded_new)))]
test_data_enc_new = df_encoded_new[int(0.7*(len(df_encoded_new))):]

target = "Weekly_Sales"
used_cols2 = [c for c in df_encoded_new.columns.to_list() if c not in [target]] # all columns except price

X_train_enc1 = train_data_enc_new[used_cols2]
X_test_enc1 = test_data_enc_new[used_cols2]
y_train_enc1 = train_data_enc_new[target]
y_test_enc1 = test_data_enc_new[target]

#droping date from train-test
X_train_enc1 = X_train_enc1.drop(['Date'], axis=1)
X_test_enc1= X_test_enc1.drop(['Date'], axis=1)
```


```python
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=40,
                           max_features = 'log2',min_samples_split = 10)

scaler=RobustScaler()
pipe = make_pipeline(scaler,rf)

pipe.fit(X_train_enc1, y_train_enc1)

# predictions on train set
y_pred_enc = pipe.predict(X_train_enc1)

# predictions on test set
y_pred_test_enc = pipe.predict(X_test_enc1)
```


```python
pipe.score(X_test_enc1,y_test_enc1)
```




    0.7397036882771106




```python
wmae_test(y_test_enc1, y_pred_test_enc)
```




    1801.5211888667177



I found best results with doing feature selection from whole encoded dataset.

# Model with Dropping Month Column

With the same dateset before, I try to model again without month column. 


```python
df_encoded_new1 = df_encoded.copy()
df_encoded_new1.drop(drop_col, axis=1, inplace=True)
```


```python
df_encoded_new1 = df_encoded_new1.drop(['Date'], axis=1)
```


```python
df_encoded_new1 = df_encoded_new1.drop(['month'], axis=1)
```


```python
#train-test split
train_data_enc_new1 = df_encoded_new1[:int(0.7*(len(df_encoded_new1)))]
test_data_enc_new1 = df_encoded_new1[int(0.7*(len(df_encoded_new1))):]

target = "Weekly_Sales"
used_cols3 = [c for c in df_encoded_new1.columns.to_list() if c not in [target]] # all columns except price

X_train_enc2 = train_data_enc_new1[used_cols3]
X_test_enc2 = test_data_enc_new1[used_cols3]
y_train_enc2 = train_data_enc_new1[target]
y_test_enc2 = test_data_enc_new1[target]
```


```python
#modeling part
pipe = make_pipeline(scaler,rf)

pipe.fit(X_train_enc2, y_train_enc2)

# predictions on train set
y_pred_enc = pipe.predict(X_train_enc2)

# predictions on test set
y_pred_test_enc = pipe.predict(X_test_enc2)
```


```python
pipe.score(X_test_enc2,y_test_enc2)
```




    0.7040151411780626




```python
wmae_test(y_test_enc2, y_pred_test_enc)
```




    2093.074731200826



It did not give better results than before.


```python
df_results = pd.DataFrame(columns=["Model", "Info",'WMAE']) # result df for showing results together
```


```python
# writing results to df
df_results = df_results.append({     
     "Model": 'RandomForestRegressor' ,
      "Info": 'w/out divided holiday columns' , 
       'WMAE' : 5850}, ignore_index=True)
```


```python
df_results = df_results.append({     
     "Model": 'RandomForestRegressor' ,
      "Info": 'w/out month column' , 
       'WMAE' : 5494}, ignore_index=True)
df_results = df_results.append({     
     "Model": 'RandomForestRegressor' ,
      "Info": 'whole data' , 
       'WMAE' : 2450}, ignore_index=True)
df_results = df_results.append({     
     "Model": 'RandomForestRegressor' ,
      "Info": 'whole data with feature selection' , 
       'WMAE' : 1801}, ignore_index=True)
df_results = df_results.append({     
     "Model": 'RandomForestRegressor' ,
      "Info": 'whole data with feature selection w/out month' , 
       'WMAE' : 2093}, ignore_index=True)
```


```python
df_results
```








The best results belongs to whole data set with feature selection. Now, I will try time series models.

# Time Series Models


```python
df.head() # to see my data
```









```python
df["Date"] = pd.to_datetime(df["Date"]) #changing data to datetime for decomposing
```


```python
df.set_index('Date', inplace=True) #seting date as index
```

# Plotting Sales


```python
plt.figure(figsize=(16,6))
df['Weekly_Sales'].plot()
plt.show()
```


    
    


In this data, there are lots of same data values. So, I will collect them together as weekly.


```python
df_week = df.resample('W').mean() #resample data as weekly
```


```python
plt.figure(figsize=(16,6))
df_week['Weekly_Sales'].plot()
plt.title('Average Sales - Weekly')
plt.show()
```


    
    


With the collecting data as weekly, I can see average sales clearly. To see monthly pattern , I resampled my data to monthly also.


```python
df_month = df.resample('MS').mean() # resampling as monthly
```


```python
plt.figure(figsize=(16,6))
df_month['Weekly_Sales'].plot()
plt.title('Average Sales - Monthly')
plt.show()
```


    
    


When I turned data to monthly, I realized that I lost some patterns in weekly data. So, I will continue with weekly resampled data.

# To Observe 2-weeks Rolling Mean and Std

My data is non-stationary. So, I will try to find more stationary version on it. 


```python
# finding 2-weeks rolling mean and std
roll_mean = df_week['Weekly_Sales'].rolling(window=2, center=False).mean()
roll_std = df_week['Weekly_Sales'].rolling(window=2, center=False).std()
```


```python
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df_week['Weekly_Sales'], color='blue',label='Average Weekly Sales')
ax.plot(roll_mean, color='red', label='Rolling 2-Week Mean')
ax.plot(roll_std, color='black', label='Rolling 2-Week Standard Deviation')
ax.legend()
fig.tight_layout()
```


    
    


# Adfuller Test to Make Sure


```python
adfuller(df_week['Weekly_Sales'])
```




    (-5.927107223737571,
     2.4290492082042356e-07,
     4,
     138,
     {'1%': -3.47864788917503,
      '5%': -2.882721765644168,
      '10%': -2.578065326612056},
     2261.596421168073)



From test and my observations my data is not stationary. So, I will try to find more stationary version of it.

# Train - Test Split of Weekly Data

To take train-test splits continuosly, I split them manually, not random.


```python
train_data = df_week[:int(0.7*(len(df_week)))] 
test_data = df_week[int(0.7*(len(df_week))):]

print('Train:', train_data.shape)
print('Test:', test_data.shape)
```

    Train: (100, 21)
    Test: (43, 21)



```python
target = "Weekly_Sales"
used_cols = [c for c in df_week.columns.to_list() if c not in [target]] # all columns except price

# assigning train-test X-y values

X_train = train_data[used_cols]
X_test = test_data[used_cols]
y_train = train_data[target]
y_test = test_data[target]
```


```python
train_data['Weekly_Sales'].plot(figsize=(20,8), title= 'Weekly_Sales', fontsize=14)
test_data['Weekly_Sales'].plot(figsize=(20,8), title= 'Weekly_Sales', fontsize=14)
plt.show()
```


    
    


Blue line represents my train data, yellow is test data.

# Decomposing Weekly Data to Observe Seasonality


```python
decomposed = decompose(df_week['Weekly_Sales'].values, 'additive', m=20) #decomposing of weekly data 
```


```python
decomposed_plot(decomposed, figure_kwargs={'figsize': (16, 10)})
plt.show()
```


    
    


From the graphs above, every 20 step seasonality converges to beginning point. This helps me to tune my model.

# Trying To Make Data More Stationary

Now, I will try to make my data more stationary. To do this, I will try model with differenced, logged and shifted data.

## 1. Difference


```python
df_week_diff = df_week['Weekly_Sales'].diff().dropna() #creating difference values
```


```python
# taking mean and std of differenced data
diff_roll_mean = df_week_diff.rolling(window=2, center=False).mean()
diff_roll_std = df_week_diff.rolling(window=2, center=False).std()
```


```python
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df_week_diff, color='blue',label='Difference')
ax.plot(diff_roll_mean, color='red', label='Rolling Mean')
ax.plot(diff_roll_std, color='black', label='Rolling Standard Deviation')
ax.legend()
fig.tight_layout()
```


    
    


## 2.Shift


```python
df_week_lag = df_week['Weekly_Sales'].shift().dropna() #shifting the data 
```


```python
lag_roll_mean = df_week_lag.rolling(window=2, center=False).mean() 
lag_roll_std = df_week_lag.rolling(window=2, center=False).std()
```


```python
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df_week_lag, color='blue',label='Difference')
ax.plot(lag_roll_mean, color='red', label='Rolling Mean')
ax.plot(lag_roll_std, color='black', label='Rolling Standard Deviation')
ax.legend()
fig.tight_layout()
```


    
    


## 3.Log


```python
logged_week = np.log1p(df_week['Weekly_Sales']).dropna() #taking log of data
```


```python
log_roll_mean = logged_week.rolling(window=2, center=False).mean()
log_roll_std = logged_week.rolling(window=2, center=False).std()
```


```python
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(logged_week, color='blue',label='Logged')
ax.plot(log_roll_mean, color='red', label='Rolling Mean')
ax.plot(log_roll_std, color='black', label='Rolling Standard Deviation')
ax.legend()
fig.tight_layout()
```


    
    


# Auto-ARIMA MODEL

I tried my data without any changes, then tried with shifting, taking log and difference version of data. Differenced data gave best results. So, I decided to take difference and use this data. 

# Train-Test Split


```python
train_data_diff = df_week_diff [:int(0.7*(len(df_week_diff )))]
test_data_diff = df_week_diff [int(0.7*(len(df_week_diff ))):]
```


```python
# train_data = train_data['Weekly_Sales']
# test_data = test_data['Weekly_Sales']

model_auto_arima = auto_arima(train_data_diff, trace=True,start_p=0, start_q=0, start_P=0, start_Q=0,
                  max_p=20, max_q=20, max_P=20, max_Q=20, seasonal=True,maxiter=200,
                  information_criterion='aic',stepwise=False, suppress_warnings=True, D=1, max_D=10,
                  error_action='ignore',approximation = False)
model_auto_arima.fit(train_data_diff)
```

     ARIMA(0,0,0)(0,0,0)[1] intercept   : AIC=1826.858, Time=0.05 sec
     ARIMA(0,0,1)(0,0,0)[1] intercept   : AIC=1793.619, Time=0.03 sec
     ARIMA(0,0,2)(0,0,0)[1] intercept   : AIC=1795.532, Time=0.08 sec
     ARIMA(0,0,3)(0,0,0)[1] intercept   : AIC=inf, Time=0.09 sec
     ARIMA(0,0,4)(0,0,0)[1] intercept   : AIC=inf, Time=0.20 sec
     ARIMA(0,0,5)(0,0,0)[1] intercept   : AIC=inf, Time=0.12 sec
     ARIMA(1,0,0)(0,0,0)[1] intercept   : AIC=1804.051, Time=0.01 sec
     ARIMA(1,0,1)(0,0,0)[1] intercept   : AIC=inf, Time=0.07 sec
     ARIMA(1,0,2)(0,0,0)[1] intercept   : AIC=1794.966, Time=0.06 sec
     ARIMA(1,0,3)(0,0,0)[1] intercept   : AIC=inf, Time=0.13 sec
     ARIMA(1,0,4)(0,0,0)[1] intercept   : AIC=inf, Time=0.18 sec
     ARIMA(2,0,0)(0,0,0)[1] intercept   : AIC=1801.215, Time=0.02 sec
     ARIMA(2,0,1)(0,0,0)[1] intercept   : AIC=inf, Time=0.12 sec
     ARIMA(2,0,2)(0,0,0)[1] intercept   : AIC=inf, Time=0.13 sec
     ARIMA(2,0,3)(0,0,0)[1] intercept   : AIC=inf, Time=0.23 sec
     ARIMA(3,0,0)(0,0,0)[1] intercept   : AIC=1791.045, Time=0.02 sec
     ARIMA(3,0,1)(0,0,0)[1] intercept   : AIC=1787.198, Time=0.05 sec
     ARIMA(3,0,2)(0,0,0)[1] intercept   : AIC=1782.922, Time=0.04 sec
     ARIMA(4,0,0)(0,0,0)[1] intercept   : AIC=1785.231, Time=0.03 sec
     ARIMA(4,0,1)(0,0,0)[1] intercept   : AIC=1786.221, Time=0.09 sec
     ARIMA(5,0,0)(0,0,0)[1] intercept   : AIC=1784.877, Time=0.03 sec
    
    Best model:  ARIMA(3,0,2)(0,0,0)[1] intercept
    Total fit time: 3.655 seconds





    ARIMA(maxiter=200, order=(3, 0, 2), scoring_args={},
          seasonal_order=(0, 0, 0, 1), suppress_warnings=True)




```python
y_pred = model_auto_arima.predict(n_periods=len(test_data_diff))
y_pred = pd.DataFrame(y_pred,index = test_data.index,columns=['Prediction'])
plt.figure(figsize=(20,6))
plt.title('Prediction of Weekly Sales Using Auto-ARIMA', fontsize=20)
plt.plot(train_data_diff, label='Train')
plt.plot(test_data_diff, label='Test')
plt.plot(y_pred, label='Prediction of ARIMA')
plt.legend(loc='best')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weekly Sales', fontsize=14)
plt.show()
```


    
    


I do not like the pattern of predictions so I decided to try another model.

# ExponentialSmoothing

I checked suitable Holt-Winters models according tp my data. Exponential Smooting are used when data has trend, and it flattens the trend. The damped trend method adds a damping parameter so, the trend converges to a constant value in the future. 

My difference data has some minus and zero values, so I used additive seasonal and trend instead of multiplicative. Seasonal periods are chosen from the decomposed graphs above. For tuning the model with iterations take too much time so, I changed and tried model for different parameters and found the best parameters and fitted them to model.


```python
model_holt_winters = ExponentialSmoothing(train_data_diff, seasonal_periods=20, seasonal='additive',
                                           trend='additive',damped=True).fit() #Taking additive trend and seasonality.
y_pred = model_holt_winters.forecast(len(test_data_diff))# Predict the test data

#Visualize train, test and predicted data.
plt.figure(figsize=(20,6))
plt.title('Prediction of Weekly Sales using ExponentialSmoothing', fontsize=20)
plt.plot(train_data_diff, label='Train')
plt.plot(test_data_diff, label='Test')
plt.plot(y_pred, label='Prediction using ExponentialSmoothing')
plt.legend(loc='best')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weekly Sales', fontsize=14)
plt.show()

```


    
    



```python
wmae_test(test_data_diff, y_pred)
```




    840.681060966696




```python
def wmae_test(test, pred): # WMAE for test 
    error = np.mean(np.abs(test - pred), axis=0)
    return error
```


```python
wmae_test(test_data_diff, y_pred)
```




    923.119482259168




```python
train_data_diff
```




    Date
    2010-02-14     -496.689958
    2010-02-21     -135.726238
    2010-02-28    -1299.253776
    2010-03-07     1008.889046
    2010-03-14     -364.671335
                      ...     
    2011-12-04    -5561.585658
    2011-12-11     1995.749152
    2011-12-18     1514.559926
    2011-12-25     5484.602409
    2012-01-01   -10088.283694
    Freq: W-SUN, Name: Weekly_Sales, Length: 99, dtype: float64



At the end, I found best results for my data with Exponential Smoothing Model.

My best result for this project is 821. According to sales amounts this value is roughly around 4-5% error. If we can take our average sales and take percentage of 821 errors, it gives 4-5% roughly. 
