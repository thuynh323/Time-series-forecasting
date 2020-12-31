import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

def error_metrics(y_fcast, y_test):
    """
    Return mean absolute percentage error (MAPE)
           mean percentage error (MPE)
           mean absolute error (MAE)
           root mean square error (RMSE)
           
    """
    print(f'MAPE: {np.mean(np.abs((y_test - y_fcast)/y_test))*100}')
    print(f'MPE:  {np.mean((y_test - y_fcast)/y_test)*100}')
    print(f'MAE:  {np.mean(np.abs(y_test - y_fcast))*100}')
    print(f'RMSE: {np.sqrt(np.mean((y_test - y_fcast)**2))}')
    

def exp_smoothing(y_train,
                  y_test,
                  trend=None,
                  seasonal=None,
                  period=None,
                  freq=None,
                  plot=False,
                  figsize=(12,5)):
    """
    Forecast using Holt-Winters exponential smoothing.
    Return a graph and error metrics.
    """
    # Modelling
    fcast_model = ExponentialSmoothing(y_train, trend=trend, seasonal=seasonal, seasonal_periods=period).fit()
    y_est = pd.DataFrame(fcast_model.fittedvalues).rename(columns={0:'y_fitted'}) # In-sample fit
    y_fcast = fcast_model.forecast(len(y_test)).rename('y_fcast') # Out-of-sample fit
    
    # Plot Series
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.set_title('Observed, Fitted, and Forecasted Series\nTriple Exponential Smoothing', fontsize=16)
        ax.set_ylabel(None)
        ax.plot(y_train, label='In-sample data', linestyle='-')
        ax.plot(y_test, label='Held-out data', linestyle='-')
        ax.plot(y_est, linestyle='--', color='g', label='Fitted values')
        ax.plot(y_fcast, linestyle='--', color='k', label='Forecasts')
        ax.legend(loc='best')
        plt.xticks(rotation = 45)
        plt.show(block = False)
        plt.close()
    
    # Print error metrics
    print('-----------------------------')
    if seasonal != None:
        print('{} trend, {} seasonality, {} {} frequency'.format(trend, seasonal, period, freq))
    error_metrics(y_fcast=y_fcast, y_test=y_test)
    print(f'AIC:  {fcast_model.aic}')
    print(f'BIC:  {fcast_model.bic}')
