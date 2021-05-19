# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:00:48 2021

@author: rathe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

data = pd.read_csv(r'C:\Users\rathe\OneDrive\Documents\python\DS.txt')

#print(yearmonth)
#print(type(data['month']))
#print(data['amount'].value_counts())

#find the index for the bad data for deletion
#r = data[data['category']=='1059']
#print(r)

#months were not consistantly cased
data['month']=data['month'].str.upper()
#a price was listed as negative. With no expert reference to ask I assumed it should be positive
data['price']=np.abs(data['price'])
#amount was occasionally miscalculated, so all values were overwritten with new calculations
data['amount'] = data['price']*data['qty']
print(data)
data=data.append({'0':'107', 'month':'APR', 'year':2018, 'category':'food', 'price':0 , 'qty':0, 'amount':0},ignore_index='true')
data=data.append({'0':'107', 'month':'JUL', 'year':2017, 'category':'dessert', 'price':0 , 'qty':0, 'amount':0},ignore_index='true')

data['datetime'] = datetime.datetime(2000, 1, 1)
for i in range(len(data)):
    data['month'][i] = datetime.datetime.strptime(data['month'][i], "%b").month
    data['datetime'][i]= datetime.datetime(data['year'][i], data['month'][i], 1)
print(data)
data.set_index('datetime')
#one row had category of '1059'
data = data.drop([55])
data = data.sort_values('datetime')

totals = data.groupby(['datetime'])['amount'].sum().reset_index()

# Plotting the Data

### Total Sales ###
fig, ax = plt.subplots(figsize=(10, 10))
# Add x-axis and y-axis
ax.plot(totals['datetime'],
        totals['amount'],
        color='blue')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Total Sales",
       title="Total Monthly Sales from 2017-2019")
plt.show()
#from the results of the graph we can see a periodic but upward trend in overall sales
#sales tend to peak in the midyear and dip 


### Food Sales ###
food_data = data[data["category"] == "food"].reset_index()
fig, ax = plt.subplots(figsize=(10, 10))
# Add x-axis and y-axis
ax.plot(food_data['datetime'],
        food_data['amount'],
        color='red')
# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Total Sales",
       title="Monthly Food Sales from 2017-2019")
plt.show()

#we can see that food sales are lowest in July

### Dessert Sales ###
dessert_data = data[data["category"] == "dessert"]
fig, ax = plt.subplots(figsize=(10, 10))
# Add x-axis and y-axis
ax.plot(dessert_data['datetime'],
        dessert_data['amount'],
        color='red')
# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Total Sales",
       title="Monthly Dessert Sales from 2017-2019")
plt.show()
#we can see that dessert sales are lowest in the midyear


### Drink Sales ###
drink_data = data[data["category"] == "drink"]
fig, ax = plt.subplots(figsize=(10, 10))
# Add x-axis and y-axis
ax.plot(drink_data['datetime'],
        drink_data['amount'],
        color='red')
# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Total Sales",
       title="Monthly Drink Sales from 2017-2019")
plt.show()
#we can see that drink sales are very predictable
#the highest sales are in the midyear and the lowest during the holidays

### Forecasting ###

import statsmodels.api as sm

# Searching for Optimal Paramters using AIC for Criterion #

#p = d = q = range(0, 2)
#pdq = list(itertools.product(p, d, q))
#seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#print('Examples of parameter for SARIMA...')
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#for param in pdq:
#   for param_seasonal in seasonal_pdq:
#        try:
#            mod = sm.tsa.statespace.SARIMAX(totals['amount'],order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
#            results = mod.fit()
#            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
#        except: 
#            continue

totals = totals.set_index(['datetime'])

total_mod = sm.tsa.statespace.SARIMAX(totals,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_invertibility=False)
results = total_mod.fit()
results.plot_diagnostics(figsize=(18, 8))
plt.show()

pred_uc = results.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = totals.plot(label='Observed', figsize=(14, 4))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Total Sales')
plt.legend()
plt.show()

### Food Data Modeling ###
food_data = data[data["category"] == "food"]
food_data = food_data[['datetime','amount']]
food_data = food_data.set_index(['datetime'])

food_mod = sm.tsa.statespace.SARIMAX(food_data,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_invertibility=False)
food_results = food_mod.fit()
food_results.plot_diagnostics(figsize=(18, 8))
plt.show()

pred_uc = food_results.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = food_data.plot(label='Observed', figsize=(14, 4))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Total Food Sales')
plt.legend()
plt.show()


### Drink Data Modeling ###
drink_data = data[data["category"] == "drink"]
drink_data = drink_data[['datetime','amount']]
drink_data = drink_data.set_index(['datetime'])
drink_mod = sm.tsa.statespace.SARIMAX(drink_data,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_invertibility=False)
drink_results = drink_mod.fit()
drink_results.plot_diagnostics(figsize=(18, 8))
plt.show()

pred_uc = drink_results.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = drink_data.plot(label='Observed', figsize=(14, 4))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Total drink Sales')
plt.legend()
plt.show()


### Dessert Data Modeling ###
dessert_data = data[data["category"] == "dessert"]
dessert_data = dessert_data[['datetime','amount']]
dessert_data = dessert_data.set_index(['datetime'])

dessert_mod = sm.tsa.statespace.SARIMAX(dessert_data,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_invertibility=False)
dessert_results = dessert_mod.fit()
dessert_results.plot_diagnostics(figsize=(18, 8))
plt.show()

pred_uc = dessert_results.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = dessert_data.plot(label='Observed', figsize=(14, 4))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Total Dessert Sales')
plt.legend()
plt.show()

### Economic Stimulus Adjustments ###

start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2020, 12, 1)

food_pred = food_results.get_prediction(start=start,end=end).predicted_mean
for i in range(4, len(food_pred)):
    food_pred[i]=(1+(-.1*(2**(4-i))))*food_pred[i]

drink_pred = drink_results.get_prediction(start=start,end=end).predicted_mean
for i in range(4, len(drink_pred)):
    drink_pred[i]=(1+(.05*(2**(4-i))))*drink_pred[i]

dessert_pred = dessert_results.get_prediction(start=start,end=end).predicted_mean
for i in range(4, len(dessert_pred)):
    dessert_pred[i]=(1+(-.05*(2**(4-i))))*dessert_pred[i]
print(dessert_pred)
### Adjustment Function ###

def stim_adj(forecast,month,year,percent):
    start = datetime.datetime(year, month, 1)
    if start > forecast.index[len(forecast)-1]:
        return[forecast]
    elif start >= forecast.index[1]:
        for i in range(month, len(forecast)):
            forecast[i]=(1+(percent*(2**(month-i))))*forecast[i]
        return[forecast]
    else:
        if forecast.index[1].year - year > 1:
            return[forecast]
        for i in range(len(forecast)):
            forecast[i]=(1+(percent*(2**-(i+month-12))))*forecast[i]
        return[forecast]