# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 21:49:53 2021

@author: shima
"""

# Importing pandas
import pandas as pd

# Importing matplotlib as plt
import matplotlib.pyplot as plt

# Import microsoft data from url naming it fpath_csv
import urllib.request

# Import numpy as np
import numpy as np

# Skewness 
# Import skew from scipy.stats
from scipy.stats import skew

# Import kurtosis from scipy.stats
from scipy.stats import kurtosis

# Import shapiro from scipy.stats
from scipy.stats import shapiro

# importing data
cum_rets = urllib.request.urlopen("https://assets.datacamp.com/production/repositories/1546/datasets/dfe9da08c986709d59943d1d5c0106537a8c608a/USO.csv")

# Calculate the running maximum
running_max = np.maximum.accumulate(cum_rets)

# Ensure the value never drops below 1
running_max[running_max < 1]= 1

# Calculate the percentage drawdown
drawdown = (cum_rets)/running_max - 1

# Plot the results
drawdown.plot()
plt.show()

StockReturns_perc = cum_rets.Returns

# Calculate historical VaR(95)
var_95 = np.percentile(StockReturns_perc, 5)
print(var_95)

# Sort the returns for plotting
sorted_rets = sorted(StockReturns_perc)

# Plot the probability of each sorted return quantile
plt.hist(sorted_rets, normed=True)

# Denote the VaR 95 quantile
plt.axvline(x=var_95, color='r', linestyle='-', label='VaR 95: {0:.2f}%'.format(var_95))
plt.show()

# Historical CVaR 95
cvar_95 = StockReturns_perc[StockReturns_perc <= var_95].mean()
print(cvar_95)

# Sort the returns for plotting
sorted_rets = sorted(StockReturns_perc)

# Plot the probability of each return quantile
plt.hist(sorted_rets, normed=True)

# Denote the VaR 95 and CVaR 95 quantiles
plt.axvline(x=var_95, color="r", linestyle="-", label='VaR 95: {0:.2f}%'.format(var_95))
plt.axvline(x=cvar_95, color='b', linestyle='-', label='CVaR 95: {0:.2f}%'.format(cvar_95))
plt.show()

# Historical VaR quantiles
var_90 = np.percentile(StockReturns_perc, 10)
print(var_90)

# Import norm from scipy.stats
from scipy.stats import norm
 
# Estimate the average daily return
mu = np.mean(StockReturns_perc)

# Estimate the daily volatility
vol = np.std(StockReturns_perc)

# Set the VaR confidence level
confidence_level = 0.05

# Calculate Parametric VaR
var_95 = norm.ppf(confidence_level, mu, vol)
print('Mean: ', str(mu), '\nVolatility: ', str(vol), '\nVaR(95): ', str(var_95))
# Historical CVaR quantiles
cvar_90 = StockReturns_perc[StockReturns_perc <= var_90].mean()
print(cvar_90)

# Plot to compare
# plot_hist()

# Aggregate forecasted VaR
forecasted_values = np.empty([100, 2])

# Loop through each forecast period
for i in range(0, 100):
    # Save the time horizon i
    forecasted_values[i, 0] = i
    # Save the forecasted VaR 95
    forecasted_values[i, 1] = var_95*np.sqrt(i+1)
    
# Plot the results
# plot_var_scale()

# Set the simulation parameters
mu = np.mean(StockReturns_perc)
vol = np.std(StockReturns_perc)
T = 252
S0 = 10

# Add one to the random returns
rand_rets = np.random.normal(mu, vol, T) + 1

# Forecasted random walk
forecasted_values = S0*rand_rets.cumprod()

# Plot the random walk
plt.plot(range(0, T), forecasted_values)
plt.show()

# Loop through 100 simulations
for i in range(100):

    # Generate the random returns
    rand_rets = np.random.normal(mu,vol,T) + 1
    
    # Create the Monte carlo path
    forecasted_values = S0*(rand_rets).cumprod()
    
    # Plot the Monte Carlo path
    plt.plot(range(T), forecasted_values)

# Show the simulations
plt.show()

# Aggregate the returns
sim_returns = []

# Loop through 100 simulations
for i in range(100):

    # Generate the Random Walk
    rand_rets = np.random.normal(mu, vol, T)
    
    # Save the results
    sim_returns.append(rand_rets)

# Calculate the VaR(99)
var_99 = np.percentile(sim_returns, 1)
print("Parametric VaR(99): ", round(100*var_99, 2),"%")


# Convert the decimal returns into percentage returns
percent_return = StockReturns_perc['Returns']*100

# Drop the missing values
returns_plot = percent_return.dropna()

# Plot the returns histogram
plt.hist(returns_plot, bins=75)
plt.show()
