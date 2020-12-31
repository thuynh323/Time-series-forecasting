def test_stationarity(y, title, window , figsize=(12,5)):
    """
    Test stationarity using moving average statistics and Dickey-Fuller test
    Source: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    """
    # Determing rolling statistics
    rolmean = y.rolling(window=window, center=False).mean()
    rolstd = y.rolling(window=window, center=False).std()
    
    # Plot rolling statistics:
    fig = plt.figure(figsize=figsize)
    orig = plt.plot(y, label='Original')
    mean = plt.plot(rolmean, color='r', label='Rolling Mean')
    std = plt.plot(rolstd, color='orange', label='Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation for ' + title, fontsize=16)
    plt.xticks(rotation = 45)
    plt.show(block = False)
    plt.close()

    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic',
                                'p-value',
                                '# Lags Used',
                                'Number of Observations Used'])
    for k, v in dftest[4].items():
        dfoutput['Critical Value (%s)'%k]=v
    print(dfoutput)
