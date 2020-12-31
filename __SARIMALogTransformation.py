import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
    
class Sarima:
    def __init__(self,
                 y_train,
                 y_test,
                 order,
                 seasonal_order):
        self.y_train = y_train
        self.y_test = y_test
        self.order = order
        self.seasonal_order = seasonal_order
        
        # Modeling
        self._model = sm.tsa.statespace.SARIMAX(self.y_train,
                                                order=self.order,
                                                seasonal_order=self.seasonal_order)
        self._results = self._model.fit()
        
        # Construct in-sample fit
        self.y_est = self._results.get_prediction()
        self.y_est_mean = self.y_est.predicted_mean
        self.y_est_ci = self.y_est.conf_int(alpha=0.05)
    
        # Construct out-of-sample forecasts
        self.y_fcast = self._results.get_forecast(steps=len(y_test)).summary_frame()
        self.y_fcast.set_index(y_test.index, inplace=True)
        
    def results(self):
        print(self._results.summary())
    
    def diagnostics(self):
        print(self._results.plot_diagnostics(figsize=(15,8)))
        
    def plot(self):
        # Transform forecast to original scale
        inv_y_fcast = np.exp(self.y_fcast)
        inv_y_est_mean = np.exp(self.y_est_mean)
        inv_y_est_ci = np.exp(self.y_est_ci)
        inv_y_train = np.exp(self.y_train)
        inv_y_test = np.exp(self.y_test)
        
        # Plot the series
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        start_index = self.order[1] + self.seasonal_order[3]
        ax.set_title('Observed, Fitted, and Forecasted Series\nSARIMA{}x{}'.format(self.order, self.seasonal_order),
                     fontsize=16)
        ax.set_ylabel(None)
        ax.plot(inv_y_train, label='In-sample data', linestyle='-')
        ax.plot(inv_y_test, label='Held-out data', linestyle='-')
        ax.plot(inv_y_est_mean[start_index :], label='Fitted values', linestyle='--', color='g')
        ax.plot(inv_y_fcast['mean'], label='Forecasts', linestyle='--', color='k')
        
        # Plot confidence intervals
        ax.fill_between(inv_y_est_mean[start_index :].index,
                        inv_y_est_ci.iloc[start_index :, 0],
                        inv_y_est_ci.iloc[start_index :, 1],
                        color='g', alpha=0.05)
        ax.fill_between(inv_y_fcast.index,
                       inv_y_fcast['mean_ci_lower'],
                       inv_y_fcast['mean_ci_upper'], 
                       color='k', alpha=0.05)
        
        ax.legend(loc='upper left')
        plt.xticks(rotation = 45)
        plt.show(block = False)
        plt.close()
        
        # Return error metrics
        error_metrics(inv_y_fcast['mean'], inv_y_test)
