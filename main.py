from datetime import datetime
from math import e, sqrt, log
from dateutil.relativedelta import relativedelta

import pandas_datareader.data as web
import scipy.stats as sci
import yfinance as yf

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


def call(spot, strike, r, t, sigma, q):
    return sci.norm.cdf(d1(spot, strike, r, t, sigma, q)) * spot * e ** (-q * t) - sci.norm.cdf(
        d2(spot, strike, r, t, sigma, q)) * strike * (
                   e ** (-r * t))


def put(spot, strike, r, t, sigma, q):
    return e ** (-r * t) * strike * sci.norm.cdf(-d2(spot, strike, r, t, sigma, q)) - spot * e ** (
            -q * t) * sci.norm.cdf(-d1(spot, strike, r, t, sigma, q))


def d1(spot, strike, r, t, sigma, q):
    return (log(spot / strike) + (r - q + sigma ** 2 / 2) * t) / (sigma * sqrt(t))


def d2(spot, strike, r, t, sigma, q):
    return d1(spot, strike, r, t, sigma, q) - sigma * sqrt(t)


# measures the rate of change of the theoretical option value with respect to changes in the underlying asset's price
def call_delta(spot, strike, r, t, sigma, q):
    return e ** (-q * t) * sci.norm.cdf(d1(spot, strike, r, t, sigma, q))


def put_delta(spot, strike, r, t, sigma, q):
    return e ** (-q * t) * -sci.norm.cdf(-d1(spot, strike, r, t, sigma, q))


# measures the rate of change in the delta with respect to changes in the underlying price
def gamma(spot, strike, r, t, sigma, q):
    return e ** (-q * t) * sci.norm.pdf(d1(spot, strike, r, t, sigma, q)) / (
            spot * sigma * sqrt(t))


# measures the sensitivity of the value of the derivative to the passage of time
def call_theta(spot, strike, r, t, sigma, q):
    return (1 / 252) * ((-(spot * sigma * e ** (-q * t) * sci.norm.pdf(d1(spot, strike, r, t, sigma, q)) / (
            2 * sqrt(t)))) - r * strike * e ** (-r * t) * sci.norm.cdf(
        d2(spot, strike, r, t, sigma, q)) + q * spot * e ** (-q * t) * sci.norm.cdf(d1(spot, strike, r, t, sigma, q)))


def put_theta(spot, strike, r, t, sigma, q):
    return (1 / 252) * ((-(spot * sigma * e ** (-q * t) * sci.norm.pdf(d1(spot, strike, r, t, sigma, q)) / (
            2 * sqrt(t)))) + r * strike * e ** (-r * t) * sci.norm.cdf(
        d2(spot, strike, r, t, sigma, q)) - q * spot * e ** (-q * t) * sci.norm.cdf(d1(spot, strike, r, t, sigma, q)))


# measures sensitivity to volatility
def vega(spot, strike, r, t, sigma, q):
    return spot * e ** (-q * t) * sqrt(t) * sci.norm.pdf(d1(spot, strike, r, t, sigma, q)) * 0.01


# measures sensitivity to the interest rate
def call_rho(spot, strike, r, t, sigma, q):
    return strike * t * e ** (-r * t) * sci.norm.cdf(d2(spot, strike, r, t, sigma, q)) * 0.01


def put_rho(spot, strike, r, t, sigma, q):
    return strike * t * e ** (-r * t) * sci.norm.cdf(-d2(spot, strike, r, t, sigma, q)) * -0.01


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
