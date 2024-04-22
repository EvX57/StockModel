import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

import preprocess

def vis_metric(df_path, metric_name, y_label, title, save_path):
    df = pd.read_csv(df_path)
    values = list(df[metric_name])
    dates = list(df['Date'])

    tick_indices = [0, len(dates)-1]
    labels = [dates[i] for i in tick_indices]

    plt.plot(values)
    plt.xlabel('Date')
    plt.xticks(ticks=tick_indices, labels=labels)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    df_path = 'Data/sp500.csv'

    vis_metric(df_path, 'Adj Close', 'Date', 'Adjusted Close Price ($)', 'S&P 500 Stock', 'Visualization/sp500.png')
