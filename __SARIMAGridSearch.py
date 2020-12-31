import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm

# Blank lists to append
mdl_index = []
mdl_aic = []
mdl_bic = []

# Set value ranges
p = range(min, max+1)
d = range(min, max+1)
q = range(min, max+1)
P = range(min, max+1)
D = range(min, max+1)
Q = range(min, max+1)
S = range(min, max+1)

# Set variables to populate
#lowest_aic = None
#lowest_parm_aic = None
#lowest_param_seasonal_aic = None

#lowest_bic = None
#lowest_parm_bic = None
#lowest_param_seasonal_bic = None

# GridSearch the hyperparameters of p, d, q and P, D, Q, S
for param in list(itertools.product(p, d, q)):
    for param_seasonal in list(itertools.product(P, D, Q, S)):
        mdl = sm.tsa.statespace.SARIMAX(log_train, order=param, seasonal_order=param_seasonal)
        results = mdl.fit()      
            # Store results
        current_aic = results.aic
        current_bic = results.bic
        mdl_index.append('SARIMA{}x{}'.format(param, param_seasonal))
        mdl_aic.append(current_aic)
        mdl_bic.append(current_bic)
            
        # Set baseline for aic
        #if lowest_aic == None:
            #lowest_aic = results.aic
        # Set baseline for bic
        #if lowest_bic == None:
            #lowest_bic = results.bic
        # Compare results
        #if current_aic <= lowest_aic:
            #lowest_aic = current_aic
            #lowest_parm_aic = param
            #lowest_param_seasonal_aic = param_seasonal
        #if current_bic <= lowest_bic:
            #lowest_bic = current_bic
            #lowest_parm_bic = param
            #lowest_param_seasonal_bic = param_seasonal            
        #print('SARIMA{}x{} - AIC:{} - BIC:{}'.format(param, param_seasonal, results.aic, results.bic))
        
#print('--------------------------------------------------------------------------------------')
#print('Model that has the lowest AIC: SARIMA{}x{} - AIC:{}'.format(lowest_parm_aic, lowest_param_seasonal_aic, lowest_aic))
#print('Model that has the lowest BIC: SARIMA{}x{} - BIC:{}'.format(lowest_parm_bic, lowest_param_seasonal_bic, lowest_bic))
