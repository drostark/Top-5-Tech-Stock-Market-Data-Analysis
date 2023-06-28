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

1.	What was the change in price of the stock over time?
2.	What was the daily return of the stock on average?
3.	What was the moving average of the various stocks?
4.	What was the correlation between different stocks closing prices?
5.	What was the correlation between different stocks daily returns?
6.	How much value do we put at risk by investing in a particular stock?
7.	How can we attempt to predict future stock behavior?
8.	How can we estimate the value at risk for a stock?

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



## Conclusion
By analyzing the market data of these top 5 tech stocks and addressing the relevant questions, investors can gain valuable insights into their performance and investment potential. The data preparation, analysis, and sharing phases provide a systematic approach to understand the stock prices, market capitalization, revenue growth, EPS, and volatility of these companies. However, it's important to conduct further research and seek professional advice before making any investment decisions.

