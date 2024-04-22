import pandas as pd
import yfinance as yf

ticker = '^GSPC'
save_path = 'Data/sp500.csv'

snp_prices = yf.download([ticker], start='2002-01-01', end='2024-04-13')
snp_prices.to_csv(save_path)