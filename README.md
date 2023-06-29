# Top 5 Tech Stock Market Data Analysis

## Introduction
In my analysis, conducted as of 28/06/2023, I will examine the market data of the top 5 tech stocks and provide insights into their performance and potential for investment. These stocks represent some of the most prominent companies in the technology sector and have shown significant growth in recent years.

## Stock List

1. **Apple Inc. (AAPL)**
   - Market Cap: $2.94 trillion
   - Interesting Points:
     - Leading global technology company known for its hardware, software, and services.
     - Strong brand recognition and customer loyalty.
     - Diversified product portfolio, including iPhone, Mac, iPad, and wearable devices.
     - Expanding services segment, including Apple Music, iCloud, and Apple Pay.

2. **Microsoft Corporation (MSFT)**
   - Market Cap: $2.49 trillion
   - Interesting Points:
     - Dominant player in the software industry, particularly with its operating system (Windows) and productivity tools (Office Suite).
     - Rapidly growing cloud computing segment with Azure platform.
     - Strong presence in the gaming industry with Xbox consoles and services.
     - Active acquisition strategy to expand product offerings.

3. **Amazon.com, Inc. (AMZN)**
   - Market Cap: $1.33 trillion
   - Interesting Points:
     - World's largest online marketplace and one of the leading cloud computing providers (Amazon Web Services).
     - Diversified business segments, including e-commerce, digital streaming, and advertising.
     - Continuous innovation with products like Amazon Echo (Alexa) and Kindle.
     - Expansion into physical retail with the acquisition of Whole Foods.

4. **Alphabet Inc. (GOOGL)**
   - Market Cap: $1.7 trillion
   - Interesting Points:
     - Parent company of Google, the dominant search engine globally.
     - Strong advertising business through Google Ads.
     - Growing presence in cloud computing with Google Cloud Platform.
     - Investments in emerging technologies, such as artificial intelligence and autonomous vehicles.

5. **Meta Platforms Inc. (META)**
   - Market Cap: $739.94 billion
   - Interesting Points:
     - Formerly known as Facebook, a leading social media platform with a large user base.
     - Diversified product ecosystem, including Facebook, Instagram, WhatsApp, and Oculus.
     - Focus on monetizing user engagement through targeted advertising.
     - Investments in virtual reality (VR) and augmented reality (AR) technologies.
    
## Dataset information

The dataset sourced from Yahoo Finance comprises pertinent information about the stocks. Below are several important columns along with their descriptions:
- Open: The opening price of the stock for a particular time period.
- High: The highest price reached by the stock during the given time period.
- Low: The lowest price reached by the stock during the given time period.
- Close: The closing price of the stock for a particular time period.
- Adj Close: The adjusted closing price of the stock, which takes into account factors such as dividends and stock splits.
- Volume: The number of shares traded for the stock during the given time period.

## Project Overview and Steps

In this project, we will employ the Google data analysis process, comprising the `Ask`, `Prepare`, `Analyze & Share` phase, to address specific inquiries related to the dataset obtained from Yahoo Finance for the top-5 stocks. Due to the unique profile of the project, the `Process`, `Analyze`, and `Share` phases are merged into the `Analyze & Share` phase, reflecting a more integrated and efficient workflow. 

## Part 1: Ask

1.	What was the change in the price of the stock over time?
2.	What was the daily return of the stock on average?
3.	What was the moving average of the various stocks?
4.	What was the correlation between different stocks' closing prices?
5.	What was the correlation between different stocks' daily returns?
6.	How much value do we put at risk by investing in a particular stock?
7.	How can we estimate the value at risk for a stock?
8.	How can we attempt to predict future stock behavior?

## Part 2: Data Preparation

To conduct this analysis, we will gather the necessary data from Yahoo Finance. The following steps were followed in the data preparation phase:
1. Imports
```python
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.stats import pearsonr
import pandas_datareader as web
import yfinance as yf
from datetime import datetime
```
2. Adjusting the plot title and incorporating interactive plotting capabilities.
```python
# Interactive plotting + title offset + style
plt.ion()
title_offset = 0.2
plt.rcParams['axes.titlepad'] = title_offset
sns.set_style('whitegrid')
```
3. Loading the data from Yahoo Finance for the previous year.
```python
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN','META']
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
for stock in tech_list:
    data = yf.download(stock, start=start, end=end)
    globals()[stock] = data
```
4. Here is a general overview of the dataset being pulled. For demonstration purposes, I will use `AAPL` (Apple Inc.) as an example to represent the dataset structure:
```python
AAPL.tail()
```
```
                  Open        High         Low       Close   Adj Close    Volume
Date
2023-06-21  184.899994  185.410004  182.589996  183.960007  183.960007  49515700
2023-06-22  183.740005  187.050003  183.669998  187.000000  187.000000  51245300
2023-06-23  185.550003  187.559998  185.009995  186.679993  186.679993  53079300
2023-06-26  186.830002  188.050003  185.229996  185.270004  185.270004  48088700
2023-06-27  185.889999  188.389999  185.669998  188.059998  188.059998  50615000
```
```python
AAPL.describe()
```
```
             Open        High         Low       Close   Adj Close        Volume
count  251.000000  251.000000  251.000000  251.000000  251.000000  2.510000e+02
mean   154.457610  156.343187  152.873028  154.743028  154.304893  7.240411e+07
std     14.227753   14.018163   14.535668   14.319082   14.424937  2.232324e+07
min    126.010002  127.769997  124.169998  125.019997  124.656982  3.519590e+07
25%    144.184998  146.650002  142.464996  144.834999  144.105446  5.598575e+07
50%    152.309998  154.259995  150.639999  152.589996  152.044266  6.874980e+07
75%    165.025002  166.385002  163.959999  165.340004  164.991791  8.161725e+07
max    186.830002  188.389999  185.669998  188.059998  188.059998  1.647624e+08
```
```python
AAPL.info()
```
```
DatetimeIndex: 251 entries, 2022-06-28 to 2023-06-27
Data columns (total 6 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Open       251 non-null    float64
 1   High       251 non-null    float64
 2   Low        251 non-null    float64
 3   Close      251 non-null    float64
 4   Adj Close  251 non-null    float64
 5   Volume     251 non-null    int64  
dtypes: float64(5), int64(1)
```
5. Nulls check
```python
AAPL.isnull().sum()
```
```
Open         0
High         0
Low          0
Close        0
Adj Close    0
Volume       0
dtype: int64
```
6. Duplicates check
```python
duplicates_count = AAPL.duplicated().sum()
print("Number of duplicated rows:", duplicates_count)
```
```
Number of duplicated rows: 0
```

## Part 3: Analyze & Share

1. To answer the first question, What was the change in price of the stock over time? I generated a line plot showing the change in the adjusted closing prices of multiple stocks over the last year. The `x-axis` represents the `dates`, and the `y-axis` represents the `adjusted close prices`. this plot allows you to visually compare the price movements of the different stocks. By examining the lines on the plot, you can observe the general trends and fluctuations in stock prices over the given time period. The differences in the slopes or peaks of the lines indicate the relative changes in price for each stock.
```python
fig, ax = plt.subplots(figsize=(10, 4))
AAPL['Adj Close'].plot(legend=True, ax=ax, label='Apple')
GOOG['Adj Close'].plot(legend=True, ax=ax, label='Google')
AMZN['Adj Close'].plot(legend=True, ax=ax, label='Amazon')
MSFT['Adj Close'].plot(legend=True, ax=ax, label='Microsoft')
META['Adj Close'].plot(legend=True, ax=ax, label='Meta Platforms')
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted Close Price')
ax.set_title('Stock Prices')
```
   ![top5_adjusted-closing-price](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/222e1afa99379254ff0cd09b67d75b061b6be4e4/Images/230628_01_top5_adjusted-closing-price.png)

Using the same approach, another line plot was created to visualize the trading volumes of the stocks over the past year. The x-axis of the plot corresponds to the dates, while the y-axis represents the trading volume.
   ![top5_volume](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/70d49910ac53358a255e5fb3bb865622faa15f26/Images/230628_01_top5_volume.png)

2. To address the question regarding the average daily return of the stocks, I computed the percentage change in the `Adj Close` column, representing the `Daily Return`. The resulting plot illustrates the daily returns of the stocks and incorporates a reference line that represents the average daily return for each stock. This enables straightforward comparisons among the stocks' daily returns and facilitates the identification of stocks with relatively higher or lower average returns.

![top5_daily_return_pct](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/07ed06ec080f51a851a8e67eaba9d8ce94011082/Images/230628_02_top5_tech_daily_return_pct.png)
Due to discrepancies in the `minimum` and `maximum` values of the 'META' stock compared to other tech stocks in the generated plot, I opted to isolate the 'META' stock and further investigate the issue.
```python
meta_returns = META['Adj Close'].pct_change()
print(meta_returns.describe())
```
```
count    251.000000
mean       0.002923
std        0.035170
min       -0.245571
25%       -0.011725
50%        0.000701
75%        0.018385
max        0.232824
Name: Adj Close, dtype: float64
```
```python
meta_returns = META['Adj Close'].pct_change()
min_date = meta_returns.idxmin()
max_date = meta_returns.idxmax()
print("Minimum return date:", min_date)
print("Maximum return date:", max_date)
```
```
Minimum return date: 2022-10-27 00:00:00
Maximum return date: 2023-02-02 00:00:00
```
The plot revealed discrepancies in the minimum and maximum values of the 'META' stock compared to other tech stocks. Upon investigating, I found that on October 27, 2022, Meta's profits sharply declined during Q3, as reported by GamesIndustry.biz. This decline was a result of a 52% decrease in net income and expected growth in the cost of revenue for the Reality Labs division. Conversely, on February 2, 2023, Meta's stock experienced a significant 23% surge, making it one of the best-performing days in over a decade, according to CNBC. These external factors explain the deviations in the daily returns of the 'META' stock and indicate that they were influenced by notable financial events rather than technical issues.

To delve further into my interest in the performance of the 'META' stock over the past year, I created a kernel density estimate (KDE) plot using Seaborn (sns). This plot effectively visualizes the distribution of the daily returns of the 'META' stock during that time period. By combining a histogram with the KDE curve, the plot offers valuable insights into the shape and spread of the returns distribution. This facilitates a deeper analysis of the stock's volatility and potential patterns, contributing to a better understanding of its overall performance.
```python
sns.distplot(META['Daily Return'].dropna(), bins=100, color='purple')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
```

   ![meta_daily_return_kde](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/cf276eecbd367381178169105dfdb66c66eed411/Images/230628_02_meta_tech_daily_return_kde.png)

3. To answer the third question, "What was the moving average of the various stocks?", a plot was generated to show the 20-day moving average for each stock. The plot represents the stock prices on the y-axis and the dates on the x-axis. Each line in the plot corresponds to the 20-day moving average for a specific stock. The moving average provides a smoothed trend line that reflects the average price over the past 20 days. This plot allows for easy comparison and analysis of the moving average trends across different stocks, providing insights into the overall direction and stability of each stock's price movement over the specified time period.

![top5_ma_20](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/11abc1354e6e42d2e514e189dfd021a2b095a94d/Images/230628_03_top5_ma_20_days.png)

4. In order to investigate the correlation between the closing prices of different stocks, I conducted the following steps. Firstly, I created a DataFrame named `closing_df` to store the closing prices of the stocks. Then, I computed the daily returns of the closing prices and removed any missing values from the dataset. These initial preparations lay the groundwork for generating plots that will visually depict the correlation between the closing prices of the different stocks.
```python
closing_df = pd.DataFrame()
for symbol in tech_list:
    data = yf.download(symbol, start=start, end=end)
    closing_df[symbol] = data['Adj Close']

sorted_closing_df_columns = sorted(closing_df.columns)
sorted_closing_df = closing_df.reindex(columns=sorted_closing_df_columns)

# Getting the daily returns for the closing prices
tech_rets = closing_df.pct_change()
tech_rets.isnull().sum()
tech_rets=tech_rets.dropna()
```
To gain insights into potential correlations and patterns among the closing prices of different stocks, I created a pair grid plot. This plot utilizes scatter plots, KDE plots, and histograms to represent the data in the respective sections of the grid. By examining the relationships and distributions of the closing prices, we can better understand any potential connections or trends among the stocks.
```python
# Pair Grid - CLOSING PRICES
returns_fig = sns.PairGrid(sorted_closing_df.dropna())
# Decide for each section (upper diagonal,lower,main)
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
```

![top5_pair_grid_closingprices](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/54eccf34e652d5d8f542e89ce4334ebcd3772e2e/Images/230629_04_pairgrid_closingprices.png)

To further clarify the correlation between the closing prices of different stocks, I created a correlation plot in the form of a heatmap. The correlation matrix was calculated based on the closing prices, indicating the strength and direction of the relationships between the stocks. A mask was applied to the heatmap to display only the lower triangular portion of the matrix, avoiding duplicate and redundant information. The values of the correlations are also annotated on the heatmap, providing a visual representation of the relationships between the stocks' closing prices.
```python
# Heatmap(Correlation plot) - CLOSING PRICES
corr_matrix_closing = sorted_closing_df.corr()
mask = np.tril(np.ones_like(corr_matrix_closing, dtype=bool))
sns.heatmap(corr_matrix_closing, annot=True,mask=mask)
```
![top5_heatmap_closingprices_corr](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/54eccf34e652d5d8f542e89ce4334ebcd3772e2e/Images/230629_04_heatmap_closingprices_corr.png)

5. To analyze the correlation between the daily returns of different stocks, I've generated another pair grid plot. This plot visually presents the relationships and distributions of the daily returns using scatter plots, KDE plots, and histograms. By studying these visual representations, we can identify any correlations, patterns, or trends in the daily returns of the stocks, providing insights into their interconnectedness and potential relationships.
```python
# Pair Grid - DAILY RETURN!
returns_fig = sns.PairGrid(sorted_tech_rets)
# Decide for each section (upper diagonal,lower,main)
returns_fig.map_upper(plt.scatter, color='green')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
```

![top5_pair_grid_dailyreturn](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/09d745ee6c33be379760af68dd0187e23469e207/Images/230629_05_pairgrid_dailyreturn.png)

Additionally, I created a correlation heatmap to analyze the relationships between the daily returns of different stocks. The heatmap presents a correlation matrix, highlighting the strength and direction of these relationships.
```python
# Heatmap(Correlation plot) - DAILY RETURN!
corr_matrix_daily = sorted_tech_rets.corr()
mask = np.tril(np.ones_like(corr_matrix_daily, dtype=bool))
sns.heatmap(corr_matrix_daily, annot=True,mask=mask, cmap="warm")
```
![top5_heatmap_dailyreturn_corr](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/09d745ee6c33be379760af68dd0187e23469e207/Images/230629_05_heatmap_dailyreturn_corr.png)

6. To visually analyze the risk versus expected return of multiple stocks, a scatter plot was generated. This plot showcases the relationship between the expected return and risk for each stock.
```python
rets = sorted_tech_rets.dropna()
area = np.pi*20
plt.scatter(x=rets.mean(), y=rets.std(), s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(25, 25), textcoords='offset points', ha='right',
        va='bottom', arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.3', color='black'))
```
![top5_scatter_risk_vs_expected_returnr](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/3c5aca6bbe3ecb049dff4d8627c41032a8f185a2/Images/230629_06_scatter_risk_vs_expected_return.png)

7. To estimate the value at risk for a stock, we can utilize the Bootstrap method for risk analysis. This approach involves calculating the empirical quantile, which provides a measure of the risk value for the stock. By determining the confidence interval using the Bootstrap method, we can assess the range within which the stock's risk falls
```python
stocks = ['AAPL', 'AMZN', 'META', 'MSFT', 'GOOG']

for stock in stocks:
    quantile = rets[stock].quantile(0.05) # 95% of confidence
    print(f"The quantile for {stock} is: {quantile}") 
```
```
The quantile for AAPL is: -0.02835961704420592
The quantile for AMZN is: -0.04021439943586136
The quantile for META is: -0.042107481842461436
The quantile for MSFT is: -0.026698565688504816
The quantile for GOOG is: -0.031143394277304393
```
For instance, the 0.05 empirical quantile of daily returns for `AAPL` is at -0.028. That means that with 95% confidence, our worst daily loss will not exceed 2.8%. If we have 1 million dollar investment, our one-day 5% VaR is 0.028*1,000,000 = $28,000.

8. To predict the future behavior of a stock, I will employ the Monte Carlo method, which involves running multiple trials with random market conditions. By simulating these scenarios, I will calculate portfolio losses for each trial and analyze the aggregated results to determine the level of risk associated with the stock.

The Monte Carlo method utilizes the geometric Brownian motion (GBM), which is a type of random walk and aligns with the weak form of the efficient market hypothesis (EMH). According to this hypothesis, past price information is already incorporated into the stock price, and future price movements are conditionally independent of past movements. 
The equation for geometric Browninan motion is given by the following equation:
$$\frac{\Delta S}{S} = \mu\Delta t + \sigma \epsilon \sqrt{\Delta t}$$


It is important to note that the past price of a stock does not provide a perfect prediction of its future value. The Monte Carlo method allows us to simulate various outcomes and assess the uncertainty associated with the stock's performance.

To demonstrate the basic application of the Monte Carlo method, we will start with a few simulations using the DataFrame and the stock symbol AAPL.
```python
# The parameters
days = 365 # Time period
dt = 1/days
mu = rets.mean()['AAPL'] # Average
sigma = rets.std()['AAPL'] # Volatility of the stock
```
```python
# Monte Carlo function - takes in the starting price and number of days and using the sigma(volatility)
# and mu(average) from the daily returns
def stock_monte_carlo(start_price,days,mu,sigma):
    price = np.zeros(days)
    price[0] = start_price

    shock = np.zeros(days)
    drift = np.zeros(days)

    for x in range(1,days):
        shock[x] = np.random.normal(loc=mu*dt, scale=sigma*np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1] * (drift[x]+shock[x]))
    return price
```
I want to point out that In the Monte Carlo simulation function, the `drift` represents the expected change in the stock price based on historical data, while the `shock` accounts for the random fluctuations or volatility in the stock price. These components are combined with the previous day's stock price to calculate the next day's price, simulating the stochastic nature of stock price movements.

Once the starting price is set at $142.13, I would proceed to run the Monte Carlo simulation for a total of 10,000 iterations.
```python
for run in range(10000):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))

plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Apple')
```
![aapl_montecarlo](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/3c5aca6bbe3ecb049dff4d8627c41032a8f185a2/Images/230629_06_scatter_risk_vs_expected_return.png)

To provide an explanation for the plot above, I will generate a histogram of the final results. Quantiles will be utilized to determine the level of risk associated with the AAPL stock.

```python
# define q as 1% empirical quantile - 99% of the values should fit within our output
q = np.percentile(simulations,1)
plt.hist(simulations, bins= 200)
# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())
# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))
# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)
# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')
# Title
plt.title(u"Final price distribution for Apple Stock after %s days" % days, weight='bold')
```
![aapl_var_hist](https://github.com/drostark/Top-5-Tech-Stock-Market-Data-Analysis/blob/3c5aca6bbe3ecb049dff4d8627c41032a8f185a2/Images/230629_06_scatter_risk_vs_expected_return.png)

## Conclusion
By analyzing the market data of these top 5 tech stocks and addressing the relevant questions, investors can gain valuable insights into their performance and investment potential. The data preparation, analysis, and sharing phases provide a systematic approach to understanding the stock prices, market capitalization, revenue growth, EPS, and volatility of these companies. However, it's essential to conduct further research and seek professional advice before making any investment decisions.

