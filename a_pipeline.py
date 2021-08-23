from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
import numpy as np
from pandas import Series
import warnings
import math
from statsmodels.tsa.arima_model import ARIMA 
from sklearn.metrics import mean_squared_error

def cleaning_df(df, location, time_hr):
    city_series = df[location]
    city_series = city_series.reset_index()
    city_series.rename({location: 'Temp'}, axis=1, inplace=True) 

    city_series['date'] = pd.to_datetime(city_series['datetime']).dt.date
    city_series['time'] = pd.to_datetime(city_series['datetime']).dt.time
    city_series['time'] = city_series['time'].astype(str)
    
    city_series_fltr = city_series[city_series['time']==time_hr]
    city_series_fltr = city_series_fltr[~city_series_fltr['Temp'].isna()]
    city_series_fltr['Converter'] = -273.15
    city_series_fltr['temp_c'] = city_series_fltr['Temp']+city_series_fltr['Converter']
    city_series_fltr = city_series_fltr.drop(['time','date','Temp', 'Converter'], axis = 1)
    
    return city_series_fltr


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value) 
    return Series(diff)

def inverse_difference(history, yhat, interval=1): 
    return yhat + history[-interval]

def cleaning(df, city, time_hr, value = None):
    city_df = cleaning_df(df, city, time_hr)
    city_df['year'] = pd.to_datetime(city_df['datetime']).dt.year
    y = city_df.drop('year', axis = 1)

    z = y['temp_c']
    z = z.values
    z = z.astype('float32')
    
    if value is not None:
        z = z[:value]
        train_size = int(len(z) * 0.50)
        train, test = z[0:train_size], z[train_size:]
    else: 
        train_size = int(len(z) * 0.50)
        train, test = z[0:train_size], z[train_size:]
    return z, train, test

def rolling_forcast(train, test):
    history = [x for x in train]
    # rolling forecasts
    predictions = list()
    for i in range(1, len(test)):
      # difference data
        months_in_year = 12
        diff = difference(history, months_in_year)
        # predict
        model = ARIMA(diff, order=(3,1,3))
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        yhat = inverse_difference(history, yhat, months_in_year) 
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
        # print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
    # report performance
    rmse = math.sqrt(mean_squared_error(test[:-1], predictions)) 
    print('RMSE: %.3f' % rmse)
    pyplot.plot(test)
    pyplot.plot(predictions, color='red') 
    pyplot.show()

def main():
    warnings.filterwarnings("ignore")
    series = read_csv("DIRECTORY_OF_CSV_FILE", header=0, index_col=0, parse_dates=True,squeeze=True) 
    z, train, test = cleaning(series, "Vancouver", "13:00:00", 300)
    rolling_forcast(train, test)

if __name__ == "__main__":
    main()

