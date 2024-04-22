import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import statistics
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, BSpline
from sklearn.linear_model import LinearRegression

# Global variables for EMA calculation
w_small = 20
w_large = 50
ema_short = []
ema_large = []
ema = {20:ema_short, 50:ema_large}

# SMA / EMA helper methods
def moving_average(datapoints, average, w_small, w_large):
    small = datapoints[-w_small:]
    large = datapoints[-w_large:]
    small_avg = average(small)
    large_avg = average(large)
    return small_avg - large_avg

def simple_average(datapoints):
    return statistics.mean(datapoints)

def exponential_average(datapoints):
    ema_vals = ema[len(datapoints)]
    ema_prev = None
    if len(ema_vals) == 0:
        ema_prev = simple_average(datapoints[:-1])
    else:
        ema_prev = ema_vals[-1]
    
    factor = 2 / (len(datapoints) + 1)
    ema_cur = datapoints[-1]*factor + ema_prev*(1-factor)
    ema[len(datapoints)].append(ema_cur)
    return ema_cur

# RSI helper methods
def percent_change(vals):
    percents = []
    for i in range(len(vals) - 1):
        p = (vals[i+1] - vals[i]) / vals[i] * 100.0
        percents.append(p)
    return percents

def RSI(datapoints):
    percents = percent_change(datapoints)
    gain = []
    loss = []
    for p in percents:
        if p >= 0.0:
            gain.append(p)
            loss.append(0.0)
        else:
            gain.append(0.0)
            loss.append(p * -1.0)
    rs = statistics.mean(gain) / statistics.mean(loss)
    return 100 - (100 / (1+rs))

# SR helper methods
def smoothing_spline(datapoints):
    x = [i for i in range(len(datapoints))]
    s = len(datapoints) * len(datapoints) / 3
    spline = splrep(x, datapoints, s=s)
    y = splev(x, spline)

    return x, y

def local_optima(datapoints):
    x, y = smoothing_spline(datapoints)
    dp = np.array(y)
    max_vals = argrelextrema(dp, np.greater)
    min_vals = argrelextrema(dp, np.less)

    y_max = [y[i] for i in max_vals[0]]
    y_min = [y[i] for i in min_vals[0]]

    return min_vals[0], y_min, max_vals[0], y_max

def find_trendline(x, y):
    threshold = 0.95
    for i in range(len(x)):
        x_temp = x[i:]
        x_temp = [[x] for x in x_temp]
        y_temp = y[i:]
        model = LinearRegression().fit(x_temp, y_temp)
        rsq = model.score(x_temp, y_temp)
        if rsq > threshold:
            y_pred = model.predict(x_temp)
            return rsq, x_temp, y_pred, model
    
    return None, None, None, None

def support_resistance(datapoints, w_size=200):
    datapoints = datapoints[-w_size:]
    x, y = smoothing_spline(datapoints)
    x_min, y_min, x_max, y_max = local_optima(y)
    r_min, _, _, model_min = find_trendline(x_min, y_min)
    r_max, _, _, model_max = find_trendline(x_max, y_max)

    if model_min != None and model_max != None:
        # Compare most recent value to the trendlines
        max_threshold = model_max.predict([[x[-1]]])[0]
        min_threshold = model_min.predict([[x[-1]]])[0]
        if datapoints[-1] >= max_threshold:
            return -1
        elif datapoints[-1] <= min_threshold:
            return 1
        else:
            return 0
    else:
        return 0


# Visualization methods 
def visualize_averages(average, datapoints, w_size, name, save_path):
    vals = []
    for i in range(len(datapoints) - w_size):
        vals.append(average(datapoints[i:i+w_size]))
    ticks = [0, len(datapoints)-w_size-1]
    labels = [gl_dates[i+w_size] for i in ticks]

    plt.plot(vals)
    plt.title(name + ' (' + str(w_size) + ' Day Window)')
    plt.xlabel('Date')
    plt.ylabel(name)
    plt.xticks(ticks=ticks, labels=labels)
    plt.savefig(save_path)
    plt.close()

def visualize_averages_overlay(average, datapoints, w_small, w_large, name, save_path):
    vals_small = []
    vals_large = []
    for i in range(len(datapoints) - w_large):
        end_index = i + w_large
        vals_small.append(average(datapoints[end_index-w_small:end_index]))
        vals_large.append(average(datapoints[i:end_index]))
    ticks = [0, len(datapoints)-w_large-1]
    labels = [gl_dates[i+w_large] for i in ticks]
    plt.plot(vals_small, color='blue', label=str(w_small) + ' Days')
    plt.plot(vals_large, color='orange', label=str(w_large) + ' Days')
    plt.title('S&P 500 - ' + name)
    plt.xlabel('Date')
    plt.ylabel(name + ' Price ($)')
    plt.xticks(ticks=ticks, labels=labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_rsi(datapoints, w_size, save_path):
    vals = []
    for i in range(len(datapoints) - w_size - 1):
        vals.append(RSI(datapoints[i:i+w_size+1]))
    ticks = [0, len(datapoints)-w_size-1]
    labels = [gl_dates[i+w_size] for i in ticks]
    plt.plot(vals)
    plt.plot([70.0 for _ in range(len(vals))], label='Sell', color='red', linewidth=2.5, linestyle='dashed')
    plt.plot([30.0 for _ in range(len(vals))], label='Buy', color='orange', linewidth=2.5, linestyle='dashed')
    plt.title('RSI (' + str(w_size) + ' Day Window)')
    plt.xlabel('Time (Days)')
    plt.ylabel('RSI')
    plt.xticks(ticks=ticks, labels=labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_rs(df_path, w_size, save_folder):
    df = pd.read_csv(df_path)
    datapoints = list(df['Adj Close'])
    dates = list(df['Date'])
    datapoints = datapoints[-w_size:]
    dates = dates[-w_size:]

    # Smoothing spline
    x, y = smoothing_spline(datapoints)
    ticks = [0, len(dates)-1]
    labels = [dates[i] for i in ticks]
    plt.plot(x, datapoints, label='Actual')
    plt.plot(x, y, label='Smoothed')
    plt.title('S&P 500 Stock - Spline Smoothing')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.xticks(ticks=ticks, labels=labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + 'spline.png')
    plt.close()

    # Local optima
    x_min, y_min, x_max, y_max = local_optima(datapoints)
    plt.plot(x, y, label='Spline')
    plt.scatter(x_min, y_min, marker='x', label='Local Minima', color='orange')
    plt.scatter(x_max, y_max, marker='x', label='Local Maxima', color='red')
    plt.title('S&P 500 Stock - Local Optima')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.xticks(ticks=ticks, labels=labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + 'optima.png')
    plt.close()

    # Trendlines
    r_min, xl_min, yl_min, _ = find_trendline(x_min, y_min)
    r_max, xl_max, yl_max, _ = find_trendline(x_max, y_max)
    print('Min: ' + str(r_min))
    print('Max: ' + str(r_max))

    plt.plot(x, y, label='Spline')
    plt.plot(xl_min, yl_min, label='Support', color='orange')
    plt.plot(xl_max, yl_max, label='Resistance', color='red')
    plt.title('S&P 500 Stock - Trendlines')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.xticks(ticks=ticks, labels=labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + 'trendlines.png')
    plt.close()


# Model
def split_train_test(df, date):
    train = []
    test = []
    
    for i in range(len(df)):
        if df.at[i, 'Date'] < date:
            train.append(df.at[i, 'Adj Close'])
        else:
            test.append(df.at[i, 'Adj Close'])
    
    return train, test

def trade_RSI(train, test, save_folder):
    n_shares = 10
    value = 0
    bought = 0
    sold = 0
    trades = 0

    profits = []
    indicators = []
    for t in test:
        ind = RSI(train[-10:])
        # Sell
        if ind > 70.0:
            sold += value*t
            value = 0
            trades += 1
        # Buy
        elif ind < 30.0:
            bought += n_shares*t
            value += n_shares
            trades += 1
        
        profit = value*t + sold - bought
        profits.append(profit)
        indicators.append(ind)
        train.append(t)
    
    # Plot
    print('Profit: ' + str(profits[-1]))
    print('Trades: ' + str(trades))

    percent = profits[-1] / bought * 100

    plt.plot(profits)
    plt.suptitle('RSI Trade Strategy')
    plt.title('Percentage Gain: ' + str(round(percent, 2)) + '%')
    plt.xlabel('Time (Days)')
    plt.ylabel('Profit ($)')
    plt.xticks(ticks=gl_ticks, labels=gl_labels)
    plt.tight_layout()
    plt.savefig(save_folder + 'RSI_profit.png')
    plt.close()

    plt.plot(indicators)
    plt.plot([70.0 for _ in range(len(indicators))], label='Sell', color='red', linewidth=2.0, linestyle='dashed')
    plt.plot([30.0 for _ in range(len(indicators))], label='Buy', color='orange', linewidth=2.0, linestyle='dashed')
    plt.suptitle('RSI Trade Strategy')
    plt.xlabel('Time (Days)')
    plt.ylabel('RSI Indicator')
    plt.xticks(ticks=gl_ticks, labels=gl_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + 'RSI_ind.png')
    plt.close()

def trade_MA(train, test, average, name, save_folder):
    n_shares = 10
    value = 0
    bought = 0
    sold = 0
    trades = 0

    profits = []
    indicators = []
    for t in test:
        ind = moving_average(train, average, 20, 50)
        # Sell
        if ind < 0.0:
            sold += value*t
            value = 0.0
            trades += 1
        # Buy
        else:
            bought += n_shares*t
            value += n_shares
            trades += 1
        
        profit = value*t + sold - bought
        profits.append(profit)
        indicators.append(ind)
        train.append(t)
    
    # Plot
    print('Profit: ' + str(profits[-1]))
    print('Trades: ' + str(trades))

    percent = profits[-1] / bought * 100

    plt.plot(profits)
    plt.suptitle(name + ' Trade Strategy')
    plt.title('Percentage Gain: ' + str(round(percent, 2)) + '%')
    plt.xlabel('Time (Days)')
    plt.ylabel('Profit ($)')
    plt.xticks(ticks=gl_ticks, labels=gl_labels)
    plt.tight_layout()
    plt.savefig(save_folder + name + '_profit.png')
    plt.close()

    plt.plot(indicators)
    plt.plot([0.0 for _ in range(len(indicators))], label='Buy/Sell Threshold', color='orange', linewidth=2.0, linestyle='dashed')
    plt.suptitle(name + ' Trade Strategy')
    plt.xlabel('Time (Days)')
    plt.ylabel(name + ' Indicator')
    plt.xticks(ticks=gl_ticks, labels=gl_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + name + '_ind.png')
    plt.close()

def trade_rs(train, test, save_folder):
    n_shares = 10
    value = 0
    bought = 0
    sold = 0
    trades = 0

    profits = []
    indicators = []
    for t in test:
        ind = support_resistance(train)
        # Sell
        if ind == -1:
            sold += value*t
            value = 0
            trades += 1
        # Buy
        elif ind == 1:
            bought += n_shares*t
            value += n_shares
            trades += 1
        
        profit = value*t + sold - bought
        profits.append(profit)
        indicators.append(ind)
        train.append(t)
    
    percent = profits[-1] / bought * 100

    # Plot
    print('Profit: ' + str(profits[-1]))
    print('Trades: ' + str(trades))

    plt.plot(profits)
    plt.suptitle('Support-Resistance Trade Strategy')
    plt.title('Percentage Gain: ' + str(round(percent, 2)) + '%')
    plt.xlabel('Time (Days)')
    plt.ylabel('Profit ($)')
    plt.xticks(ticks=gl_ticks, labels=gl_labels)
    plt.tight_layout()
    plt.savefig(save_folder + 'SR_profit.png')
    plt.close()

    plt.plot(indicators)
    plt.suptitle('Support-Resistance Trade Strategy')
    plt.xlabel('Time (Days)')
    plt.ylabel('Support-Resistance Indicator')
    plt.xticks(ticks=gl_ticks, labels=gl_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + 'SR_ind.png')
    plt.close()

if __name__ == '__main__':
    df = pd.read_csv('Data/sp500.csv')
    datapoints = list(df['Adj Close'])
    gl_dates = list(df['Date'])

    #visualize_averages(exponential_average, datapoints, 50, 'EMA', 'MA/EMA_50.png')
    #visualize_averages_overlay(exponential_average, datapoints, 20, 50, 'EMA', 'MA/EMA.png')
    #visualize_rsi(datapoints, 10, 'RSI/RSI_10.png')
    #visualize_rs('Data/amazon.csv', 200, 'Trendline/')

    train, test = split_train_test(df, '2023-05-11')
    gl_ticks = [0, len(test)-1]
    gl_labels = [gl_dates[i+len(gl_dates)-len(test)] for i in gl_ticks]

    #trade_RSI(train, test, 'RSI/')
    #trade_MA(train, test, exponential_average, 'EMA', 'MA/')
    #trade_rs(train, test, 'RS 2/')