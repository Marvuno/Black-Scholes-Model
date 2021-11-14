from datetime import datetime
from math import e, sqrt, log
from dateutil.relativedelta import relativedelta

import pandas_datareader.data as web
import scipy.stats as sci
import yfinance as yf


def d1(spot, strike, r, t, sigma, q):
    return (log(spot / strike) + (r - q + sigma ** 2 / 2) * t) / (sigma * sqrt(t))


def d2(spot, strike, r, t, sigma, q):
    return d1(spot, strike, r, t, sigma, q) - sigma * sqrt(t)


def call(spot, strike, r, t, sigma, q):
    return cdf_d1 * spot * exp_div - cdf_d2 * strike * exp_r


def put(spot, strike, r, t, sigma, q):
    return exp_r * strike * cdf_negative_d2 - spot * exp_div * cdf_negative_d1


# the sensitivity of an option’s price changes relative to the changes in the underlying asset’s price
def call_delta(spot, strike, r, t, sigma, q):
    return exp_div * cdf_d1


def put_delta(spot, strike, r, t, sigma, q):
    return exp_div * -cdf_negative_d1


# the delta’s change relative to the changes in the price of the underlying asset
def gamma(spot, strike, r, t, sigma, q):
    return exp_div * pdf_d1 / (spot * sigma * sqrt(t))


# the sensitivity of the option price relative to the option’s time to maturity
def call_theta(spot, strike, r, t, sigma, q):
    return (1 / 252) * ((-(spot * sigma * exp_div * pdf_d1 / (2 * sqrt(t)))) - r * strike * exp_r * cdf_d2
                        + q * spot * exp_div * cdf_d1)


def put_theta(spot, strike, r, t, sigma, q):
    return (1 / 252) * ((-(spot * sigma * exp_div * pdf_d1 / (2 * sqrt(t)))) + r * strike * exp_r * cdf_d2
                        - q * spot * exp_div * cdf_d1)


# the sensitivity of an option price relative to the volatility of the underlying asset
def vega(spot, strike, r, t, sigma, q):
    return spot * exp_div * sqrt(t) * pdf_d1 * 0.01


# the sensitivity of the option price relative to interest rates
def call_rho(spot, strike, r, t, sigma, q):
    return strike * t * exp_r * cdf_d2 * 0.01


def put_rho(spot, strike, r, t, sigma, q):
    return strike * t * exp_r * cdf_negative_d2 * -0.01


stock = input("What is the stock code? ")
expiry = input("When is the expiry date of the option?(In mm-dd-yyyy format) ")
strike = int(input("What is the strike price? "))

treasury = "^TNX"
# stock = "TSLA"
# expiry = "11-19-2021"
# strike = 1035
dt = datetime.utcnow()
one_year_ago = dt - relativedelta(years=1)

df = web.DataReader(stock, 'yahoo', start=one_year_ago, end=dt)
df = df.sort_values(by="Date").dropna().assign(close_day_before=df.Close.shift(1))
df['returns'] = (df.Close - df.close_day_before) / df.close_day_before

spot = df['Close'].iloc[-1]
r = web.DataReader(treasury, 'yahoo', start=one_year_ago, end=dt)['Close'].iloc[-1] / 100
t = (datetime.strptime(expiry, "%m-%d-%Y") - dt).days / 252
sigma = df['returns'].std() * sqrt(252)

try:
    q = yf.Ticker(stock).dividends.iloc[-1] / spot
except IndexError:
    q = 0

exp_div = e ** (-q * t)
exp_r = e ** (-r * t)
cdf_d1 = sci.norm.cdf(d1(spot, strike, r, t, sigma, q))
cdf_d2 = sci.norm.cdf(d2(spot, strike, r, t, sigma, q))
pdf_d1 = sci.norm.pdf(d1(spot, strike, r, t, sigma, q))
pdf_d2 = sci.norm.pdf(d2(spot, strike, r, t, sigma, q))
cdf_negative_d1 = sci.norm.cdf(-d1(spot, strike, r, t, sigma, q))
cdf_negative_d2 = sci.norm.cdf(-d2(spot, strike, r, t, sigma, q))

print("-------------------------------------------------------")
print(f"Call Option Price: {call(spot, strike, r, t, sigma, q)}")
print(f"Put Option Price: {put(spot, strike, r, t, sigma, q)}")
print("-------------------------------------------------------")
print(f"Call Delta: {call_delta(spot, strike, r, t, sigma, q)}")
print(f"Put Delta: {put_delta(spot, strike, r, t, sigma, q)}")
print(f"Gamma: {gamma(spot, strike, r, t, sigma, q)}")
print(f"Call Theta: {call_theta(spot, strike, r, t, sigma, q)}")
print(f"Put Theta: {put_theta(spot, strike, r, t, sigma, q)}")
print(f"Vega: {vega(spot, strike, r, t, sigma, q)}")
print(f"Call Rho: {call_rho(spot, strike, r, t, sigma, q)}")
print(f"Put Rho: {put_rho(spot, strike, r, t, sigma, q)}")