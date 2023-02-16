from datetime import datetime, timedelta
import pytz
from math import e, sqrt, log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
from scipy.optimize import brentq
import yfinance as yf
import openpyxl


def implied_volatility(spot, strike, r, t, q, option):
    def error_function(sigma):
        return call(spot, strike, r, t, sigma, q) - option

    return brentq(error_function, a=2, b=-2)


def d1(spot, strike, r, t, sigma, q):
    return (log(spot / strike) + (r - q + sigma ** 2 / 2) * t) / (sigma * sqrt(t))


def d2(spot, strike, r, t, sigma, q):
    return d1(spot, strike, r, t, sigma, q) - sigma * sqrt(t)


def call(spot, strike, r, t, sigma, q):
    return sci.norm.cdf(d1(spot, strike, r, t, sigma, q)) * spot * (e ** (-q * t)) - sci.norm.cdf(d2(spot, strike, r, t, sigma, q)) * strike * (e ** (-r * t))


def put(spot, strike, r, t, sigma, q):
    return (e ** (-r * t)) * strike * sci.norm.cdf(-d2(spot, strike, r, t, sigma, q)) - spot * (e ** (-q * t)) * sci.norm.cdf(-d1(spot, strike, r, t, sigma, q))


# the sensitivity of an option’s price changes relative to the changes in the underlying asset’s price
def call_delta(spot, strike, r, t, sigma, q):
    return (e ** (-q * t)) * sci.norm.cdf(d1(spot, strike, r, t, sigma, q))


def put_delta(spot, strike, r, t, sigma, q):
    return (e ** (-q * t)) * sci.norm.cdf(-d1(spot, strike, r, t, sigma, q))


# the delta’s change relative to the changes in the price of the underlying asset
def gamma(spot, strike, r, t, sigma, q):
    return (e ** (-q * t)) * sci.norm.pdf(d1(spot, strike, r, t, sigma, q)) / (spot * sigma * sqrt(t))


# the sensitivity of the option price relative to the option’s time to maturity
def call_theta(spot, strike, r, t, sigma, q):
    return (1 / 252) * ((-(spot * sigma * (e ** (-q * t)) * sci.norm.pdf(d1(spot, strike, r, t, sigma, q)) / (2 * sqrt(t)))) - r * strike * (e ** (-r * t))
                        * sci.norm.cdf(d2(spot, strike, r, t, sigma, q)) + q * spot * (e ** (-q * t)) * sci.norm.cdf(d1(spot, strike, r, t, sigma, q)))


def put_theta(spot, strike, r, t, sigma, q):
    return (1 / 252) * ((-(spot * sigma * (e ** (-q * t)) * sci.norm.pdf(d1(spot, strike, r, t, sigma, q)) / (2 * sqrt(t)))) + r * strike * (e ** (-r * t))
                        * sci.norm.cdf(d2(spot, strike, r, t, sigma, q)) - q * spot * (e ** (-q * t)) * sci.norm.cdf(d1(spot, strike, r, t, sigma, q)))


# the sensitivity of an option price relative to the volatility of the underlying asset
def vega(spot, strike, r, t, sigma, q):
    return spot * (e ** (-q * t)) * sqrt(t) * sci.norm.pdf(d1(spot, strike, r, t, sigma, q)) * 0.01


# the sensitivity of the option price relative to interest rates
def call_rho(spot, strike, r, t, sigma, q):
    return strike * t * (e ** (-r * t)) * sci.norm.cdf(d2(spot, strike, r, t, sigma, q)) * 0.01


def put_rho(spot, strike, r, t, sigma, q):
    return strike * t * (e ** (-r * t)) * sci.norm.cdf(-d2(spot, strike, r, t, sigma, q)) * -0.01


def option_summary(stock: list, expiry: str):
    """
    input a list of stock tickers, and an expiry date (exact date not required). This program will generate the summary stats of the option.
    Main focus is the implied volatility and the option greeks. Data is acquired from Yahoo Finance.
    """
    treasury = "^TNX"
    data = []

    for ticker in stock:
        ticker = ticker.upper()
        dt = datetime.utcnow()

        # find the closest expiry date to user input
        try:
            expiry = datetime.strptime(min(yf.Ticker(ticker).options, key=lambda date: abs(datetime.strptime(date, "%Y-%m-%d") -
                                                                                           datetime.strptime(expiry, "%m-%d-%Y"))), "%Y-%m-%d").strftime(
                "%m-%d-%Y")
        # in case of lack of data
        except:
            continue

        spot = yf.Ticker(ticker).history(period="7d")['Close'].iloc[-1]
        # 10 yr US treasury yield as risk-free rate (one day ago)
        r = yf.Ticker(treasury).history(period="7d").Close.iloc[-1] / 100
        t = (datetime.strptime(expiry, "%m-%d-%Y") - dt).days / 252

        try:
            q = yf.Ticker(ticker).dividends.iloc[-1] / spot
        except IndexError:
            q = 0

        # get the out-of-money options with the closest strike price
        calls = yf.Ticker(ticker).option_chain(datetime.strptime(expiry, "%m-%d-%Y").strftime("%Y-%m-%d")).calls
        calls = calls[calls['inTheMoney'] == False].iloc[0]
        strike = calls.strike
        option = calls.lastPrice

        # solve by brent's method
        sigma = implied_volatility(spot, strike, r, t, q, option)

        # update US trading date at 9am Eastern time
        if datetime.now(pytz.timezone('US/Eastern')).hour >= 9:
            us_time = (datetime.now(pytz.timezone('US/Eastern'))).strftime("%m-%d-%Y")
        else:
            us_time = (datetime.now(pytz.timezone('US/Eastern')) - timedelta(days=1)).strftime("%m-%d-%Y")

        # loop in the future
        data.append(
            [ticker, us_time, expiry, round(spot, 2), strike, call(spot, strike, r, t, sigma, q),
             put(spot, strike, r, t, sigma, q), sigma * 100, call_delta(spot, strike, r, t, sigma, q),
             put_delta(spot, strike, r, t, sigma, q), gamma(spot, strike, r, t, sigma, q),
             call_theta(spot, strike, r, t, sigma, q), put_theta(spot, strike, r, t, sigma, q), vega(spot, strike, r, t, sigma, q),
             call_rho(spot, strike, r, t, sigma, q), put_rho(spot, strike, r, t, sigma, q)])

    df = pd.DataFrame(data, columns=['Ticker', 'Today Date', 'Expiry Date', 'Spot', 'Strike', 'Call Option', 'Put Option', 'Implied Volatility(%)',
                                     'Call Delta', 'Put Delta', 'Gamma', 'Call Theta', 'Put Theta', 'Vega', 'Call Rho', 'Put Rho'])
    return df


def option_sentiment(stock: str):
    """
    Get price change of option at different expiry dates for the same stock to determine if investors sentiment is reflected
    """
    data = []
    ticker = stock.upper()
    expiry_dates = yf.Ticker(ticker).options

    # update US trading date at 9am Eastern time
    if datetime.now(pytz.timezone('US/Eastern')).hour >= 9:
        us_time = (datetime.now(pytz.timezone('US/Eastern'))).strftime("%Y-%m-%d")
    else:
        us_time = (datetime.now(pytz.timezone('US/Eastern')) - timedelta(days=1)).strftime("%Y-%m-%d")

    spot = yf.Ticker(ticker).history(period="7d")['Close'].iloc[-1]
    data.append([us_time, spot])
    # print(f"{datetime.today().strftime('%Y-%m-%d')}: {round(spot, 2)}")

    for date in expiry_dates:
        calls = yf.Ticker(ticker).option_chain(date).calls
        calls = calls.fillna(0)
        avg_price = sum(calls.strike * calls.volume) / sum(calls.volume)
        data.append([date, avg_price])
        # print(f"{date}: {round(avg_price, 2)}")

    df = pd.DataFrame(data, columns=['Date', 'Price'])
    return df


def plot_option_vol_skew(stock: str):
    """
    Plotting Call Implied Volatility Surface for quant trading
    """
    data = []
    treasury = "^TNX"
    ticker = yf.Ticker(stock.upper())

    # update US trading date at 9am Eastern time
    if datetime.now(pytz.timezone('US/Eastern')).hour >= 9:
        us_time = (datetime.now(pytz.timezone('US/Eastern'))).strftime("%Y-%m-%d")
    else:
        us_time = (datetime.now(pytz.timezone('US/Eastern')) - timedelta(days=1)).strftime("%Y-%m-%d")

    expiry_list = ticker.options
    days_of_expiration = [(datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(us_time, "%Y-%m-%d")).days for date in expiry_list]

    spot = ticker.history(period="7d")['Close'].iloc[-1]
    # 10 yr US treasury yield as risk-free rate (one day ago)
    r = yf.Ticker(treasury).history(period="7d").Close.iloc[-1] / 100

    try:
        q = ticker.dividends.iloc[-1] / spot
    except IndexError:
        q = 0

    # different expiry dates
    for n in range(len(expiry_list)):
        t = (datetime.strptime(expiry_list[n], "%Y-%m-%d") - datetime.strptime(us_time, "%Y-%m-%d")).days / 252
        strike_list = list(ticker.option_chain(expiry_list[n])[0].strike)
        option_list = list(ticker.option_chain(expiry_list[n])[0].lastPrice)

        # update strike and option price
        for i in range(len(strike_list)):

            strike = strike_list[i]
            option = option_list[i]

            # solve by brent's method
            try:
                sigma = implied_volatility(spot, strike, r, t, q, option) * 100
            except ValueError:
                continue

            # neglect extreme case
            if sigma >= 0.1:
                data.append([days_of_expiration[n], strike, sigma])

    df = pd.DataFrame(data, columns=['Expiration Days', 'Strike', 'Implied Volatility'])

    # plotting call implied volatility surface
    surface = df.pivot_table(values="Implied Volatility", index="Strike", columns="Expiration Days").dropna()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(surface.columns.values, surface.index.values)
    z = surface.values
    ax.set_xlabel('Days to Expiration')
    ax.set_ylabel('Strike Price')
    ax.set_zlabel('Implied Volatility (%)')
    ax.set_title(f"Call Implied Volatility Surface: {stock}")
    ax.plot_surface(x, y, z)
    plt.show()

    return df


print(option_summary(["TSLA"], "03-10-2023").to_string())
print(option_sentiment("TSLA").to_string())
print(plot_option_vol_skew("TSLA").to_string())

# # Get data for top 100 US stocks by market cap
# tickers = list(pd.read_excel('Constituents.xlsx', index_col=0)['Symbol'][:100])
# option_summary(tickers, "03-10-2023").to_csv('options.csv', index=False)
