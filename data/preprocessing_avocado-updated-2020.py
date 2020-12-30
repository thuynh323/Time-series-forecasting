import pandas as pd
import datetime as dt

data = pd.read_csv('avocado-updated-2020.csv')
my_data = data[(data['geography'] == 'Total U.S.') &
               (data['type'] == 'organic')][['date', 'total_volume']]
my_data = my_data[(my_data['date'].dt.year != 2019) &
                  (my_data['date'].dt.year != 2020)].set_index('date')
my_data.drop(pd.Timestamp('2018-01-01'), inplace=True)
my_data.to_csv('organic_avocado.csv', index_label=False)
