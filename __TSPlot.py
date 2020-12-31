import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt

def plot_general(y, title='title', lags=None, figsize=(12,8)):
    """
    Examine the patterns of ACF and PACF, along with the time series plot and histogram.
    Source: https://github.com/jeffrey-yau/Pearson-TSA-Training-Beginner/blob/master/1_Intro_and_Overview.ipynb
    """
    fig = plt.figure(figsize=figsize)
    layout = (2,2)
    ts_ax = plt.subplot2grid(layout, (0,0))
    hist_ax = plt.subplot2grid(layout, (0,1))
    acf_ax = plt.subplot2grid(layout, (1,0))
    pacf_ax = plt.subplot2grid(layout, (1,1))
    
    y.plot(ax=ts_ax)
    ts_ax.set_xlabel(None)
    ts_ax.set_title(title)
    
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    sns.despine()
    plt.tight_layout()
