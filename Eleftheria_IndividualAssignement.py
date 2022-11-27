# -*- coding: utf-8 -*-
###############################################################################
# FINANCIAL DASHBOARD
###############################################################################

# ==============================================================================
# Initiating
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from datetime import date
import yfinance as yf
import streamlit as st

# ==============================================================================
# Summary
# ==============================================================================


@st.cache
def GetStockData(period, interval):
    global ticker
    stock_price = pd.DataFrame()
    # Get the data for the selected period
    stock_df = yf.Ticker(ticker).history(period, interval)
    stock_df['Ticker'] = ticker  # Add the column ticker name
    stock_price = pd.concat([stock_price, stock_df],
                            axis=0)  # Combine results
    return stock_price.loc[:, ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Function that creates the line plot to display in the summary


def ShowLinePlot():
    # Get the ticker (global variable)
    global ticker
    # Get the stock_price (global variable)
    global stock_price
    # Get the values from the stock_price only for the selected ticker
    stock_df = stock_price[stock_price['Ticker'] == ticker]

    # Create the figure to print the data and give it a title
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=(f'{ticker} Stock Price'),
    )

    # Add a trace (Scatter) which will print the data in the figure
    fig.add_trace(
        go.Scatter(
            # The X axis will be the dataframe index (the dates)
            x=stock_df.index,
            # The Y axis will be the close price from the dataframe
            y=stock_df['Close'],
            # Give a name to the close price line that we are printing for better understanding
            name='Close price',
            # Fill the space underneath the line to make it look similar to the Yahoo graph
            fill="tozeroy",
            line_color='green'  # Select the color that the line should be
        ),
        row=1,  # Print it in the row 1 (the figure only has 1 row)
        col=1,  # Print it in the column 1 (the figure only has 1 column)
    )

    # Update the layout to remove the grids
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    # Print the frigure
    st.plotly_chart(fig)

# Function that has all the functionality for the summary tab


def summary():

    # Cache the call to the company info (not needed more than once)
    @st.cache
    def GetCompanyInfo():
        global ticker
        return yf.Ticker(ticker).info

    # Cache the call to the major holders (not needed more than once)
    @st.cache
    def GetCompanyMajorHolders():
        global ticker
        return yf.Ticker(ticker).major_holders

    # Check if the user has selected any tickers in the sidebar
    if ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo()

        # Add dashboard title and description
        st.title(f"Summary of {info['shortName']}")
        st.subheader(f"{info['longName']}")

        # Create two columns to display the different data from the company
        summ1, summ2 = st.columns(2)
        # On the left column print some data from the company
        with summ1:
            st.write(f"{info['address1']}")
            st.write(f"{info['city']}, {info['state']} {info['zip']}")
            st.write(f"{info['country']}")
            st.write(f"{info['phone']}")
            st.write(f"{info['website']}")
        # On the right column print the rest of the data
        with summ2:
            st.write(f"Sector(s): **{info['sector']}**")
            st.write(f"Industry: **{info['industry']}**")
            st.write(
                f"Full Time Employees: **{'{:,}'.format(info['fullTimeEmployees'])}**")

        # Print the company description
        st.subheader("Description")
        st.write(info['longBusinessSummary'])

        st.subheader("Stats")
        # Create two columns to print the stats of each company
        stats1, stats2 = st.columns(2)

        # On the left column print as shown in Yahoo stats some of the values
        with stats1:
            st.metric(label="Previous Close", value=f"{info['previousClose']}")
            st.metric(label="Open", value=f"{info['open']}")
            st.metric(label="Bid", value=f"{info['bid']}")
            st.metric(label="Ask ", value=f"{info['ask']}")
            # Print Day's Range
            st.metric(label="Day's Range",
                      value=f"{info['dayLow']} - {info['dayHigh']}")
            # Print Volume with number format (add , to miles)
            st.metric(label="Volume ",
                      value=f"{'{:,}'.format(info['volume'])}")
            # Print Avg. Volume with number format (add , to miles)
            st.metric(label="Avg. Volume ",
                      value=f"{'{:,}'.format(info['averageVolume'])}")

        # On the right column print as shown in Yahoo stats the rest of the values
        with stats2:
            # Print Market Cap with number format (add , to thousands)
            st.metric(
                label="Market Cap",  # Give a label to the metric
                # Print the actual value of the metric (formatted)
                value=f"{'{:,}'.format(info['marketCap'])}"
            )
            st.metric(
                label="Beta (5Y Monthly)",
                value=f"{info['beta']}"
            )
            st.metric(
                label="EPS (TTM)",
                value=f"{info['trailingEps']}"
            )
            # Create datetime from the timestamp to format it if the value is not none
            if info['exDividendDate'] != None:
                date_time = datetime.fromtimestamp(info['exDividendDate'])
                st.metric(
                    label="Ex-Dividend Date",
                          value=f"{date_time.strftime('%b %d, %Y')}"
                )
        # Create a slider with the different periods of time that the user can select (filtering the data) to create the graph
        period = st.select_slider(
            'Select a period of time',
            options=['1mo', '3mo', '6mo', 'ytd', '1y',
                     '2y', '5y', '10y', 'max']
        )

        global stock_price
        # Get the data from yfinance with the selected period of time, and interval of 1d as specified by the task
        stock_price = GetStockData(period, '1d')
        # Display the area plot
        ShowLinePlot()

        # Display the major shareholders from the company
        st.subheader("Major Shareholders")
        st.dataframe(
            GetCompanyMajorHolders()
        )


# ==============================================================================
# Chart
# ==============================================================================

# Calculate the Moving Average (MA) by using the functionalities provided by pandas.
# Rolling is function that helps us to make calculations on a rolling window.
# On the rolling window, we will use .mean() function to calculate the mean of each window (50 days as stablished by the task)
def GetMAFiftyDays(df):
    return df['Close'].rolling(50).mean()

# Function that displays the line chart
def ShowLineChart():
    global ticker
    global stock_price
    st.write('Close price')
    # Create the figure to print the data on
    fig, ax = plt.subplots(figsize=(5, 5))

    # Get the data only from the selected ticker by the user
    stock_df = stock_price[stock_price['Ticker'] == ticker]

    # Create the layout for the plots to display
    fig = make_subplots(specs=[[{"secondary_y": True}]]
    )

    # Add the first trace (plot) which in this case will be the line with the close price
    fig.add_trace(
        # Plot the line with the close price (Y Axis) and the index (X Axis)
        go.Line(x=stock_df.index, y=stock_df['Close'], name='Close price')
    )

    # Append another trace (plot) to display the MA on the the same top plot
    fig.add_trace(
        # Plot the line with the MA (Y Axis) and the index (X Axis)
        go.Line(
            x=stock_df.index,
            y=GetMAFiftyDays(stock_df),
            name='MA 50',
            line=dict(color="#ffe476")
        )
    )

    # Add the second plot, which will be a bar plot with the total volume
    fig.add_trace(
        # Plot the bars with the Volume (Y Axis) and the index (X Axis)
        go.Bar(x=stock_df.index, y=stock_df['Volume'], name='Volume', opacity = 0.5),
        secondary_y=True
    )

    # Give the axis titles for better understanding of the charts
    fig['layout']['xaxis']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'

    fig.update_xaxes(
        rangebreaks = [{'bounds': ['sat', 'mon']}],
        rangeslider_visible = False
    )
    # Print the figure
    st.plotly_chart(fig)


# Function that displays the candle chart
def ShowCandleChart():
    global stock_price
    # Get the data only from the selected ticker by the user
    stock_df = stock_price[stock_price['Ticker'] == ticker]

    # Create the layout for the plots to display
    fig =  make_subplots(specs=[[{"secondary_y": True}]])

    # Add the trace (plot) which in this case will be a candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_df.index,  # X Axis will be the date
            # Add the different values needed to interpret a candlestick chart
            open=stock_df['Open'],
            high=stock_df['High'],
            low=stock_df['Low'],
            close=stock_df['Close'],
            # Give the chart candles a name
            name='Candlestick chart'
        ),
    )

    # Append another trace (plot) to display the MA on the the same top plot
    fig.add_trace(
        # Plot the line with the MA (Y Axis) and the index (X Axis)
        go.Line(
            x=stock_df.index,
            y=GetMAFiftyDays(stock_df),
            name='MA 50',
            line=dict(color="#ffe476")
        ),
    )

    # Add the second plot, which will be a bar plot with the total volume
    fig.add_trace(
        # Plot the bars with the Volume (Y Axis) and the index (X Axis)
        go.Bar(x=stock_df.index, y=stock_df['Volume'], name='Volume', opacity=0.5),
        secondary_y=True
    )

    # Give the axis titles for better understanding of the charts
    fig['layout']['xaxis']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'


    fig.update_xaxes(
        rangebreaks=[{'bounds': ['sat', 'mon']}],
        rangeslider_visible=False,
    )

    # Print the figure
    st.plotly_chart(fig)
    
    
# Function that has all the functionality for the chart tab
def chart():

    # Add dashboard title and description
    st.title("Chart")
    # Create three columns with the options for the user
    select1, select2, select3 = st.columns(3)
    with select1:
        # In the first columns add the options to select the period of time
        period = st.select_slider(
            'Select a period of time',
            options=['1mo', '3mo', '6mo', 'ytd', '1y',
                     '2y', '5y', '10y', 'max']
        )
    with select2:
        # Create interval selector (we can't go further than 3months, specified in the documentation of yfinance)
        # Invalid input - interval=1y is not supported. Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
        interval = st.select_slider(
            'Select interval',
            options=['1d', '1wk', '1mo', '3mo']
        )
    with select3:
        # Add and option to allow the user to switch between the Line cahrt and the Candlestick chart
        chart_type = st.radio(
            "Set type of chart",
            ["Line plot", "Candle plot"],
            horizontal=True,
        )

    global stock_price
    # Get the data that should be displayed in the charts
    stock_price = GetStockData(period, interval)

    # Based on the user selection, display one chart or the other one
    if chart_type == 'Line plot':
        ShowLineChart()
    else:
        ShowCandleChart()

# ==============================================================================
# Financial
# ==============================================================================


@st.cache
# Get the data from the financials, and cache (stores it locally) it to avoid multiple calls to the yfinance library
# This function receives two parameters:
# - Show: which specifies which tab the user wants to see -> Income Statement, Balance Sheet, Cash Flow
# - Period: specifies if the data shown should be annual or quarterly
def GetCompanyFinancials(show, period):
    global ticker
    if show == 'Income Statement':
        if period == 'Anual':
            financials = yf.Ticker(ticker).financials
        else:
            financials = yf.Ticker(ticker).quarterly_financials
    elif show == 'Balance Sheet':
        if period == 'Anual':
            financials = yf.Ticker(ticker).balance_sheet
        else:
            financials = yf.Ticker(ticker).quarterly_balance_sheet
    elif show == 'Cash Flow':
        if period == 'Anual':
            financials = yf.Ticker(ticker).cashflow
        else:
            financials = yf.Ticker(ticker).quarterly_cashflow

    return financials

# Function that has all the functionality for the financials tab
def financial():

    # Add dashboard title and description
    st.title("Financials")
    # Create two columns with the options to allow the user to select which data and period they want to see
    select1, select2 = st.columns(2)
    with select1:
        # Add a radio selector to allow the user to select the data they want to see
        show = st.radio(
            "Show:",
            ["Income Statement", "Balance Sheet", "Cash Flow"],
            horizontal=True,
        )
    with select2:
        # Add a radio selector to allow the user to select the period of data they want to see
        period = st.radio(
            "",
            ["Anual", "Quarterly"],
            horizontal=True,
        )

    # Display the data
    st.write(GetCompanyFinancials(show, period))
# ==============================================================================
# Monte Carlo simulation
# ==============================================================================

# Object that has the implementation done by the professor Minh Phan in class with the methods to:
# - Initiate the object
# - Run the simulation 
# - Print the data from the simulation
# Little tweaks have been made to adapt it to our script (Use of the yfinance library methods instead of the ones used by the teacher)
class MonteCarlo(object):

    def __init__(self, ticker, start_date, end_date, time_horizon, n_simulation, seed):

        # Initiate class variables
        self.ticker = ticker  # Stock ticker
        self.start_date = start_date  # Text, YYYY-MM-DD
        self.end_date = end_date  # Text, YYYY-MM-DD
        self.time_horizon = time_horizon  # Days
        self.n_simulation = n_simulation  # Number of simulations
        self.seed = seed  # Random seed
        self.simulation_df = pd.DataFrame()  # Table of results

        # Extract stock data
        self.stock_price = yf.Ticker(ticker).history(
            start=self.start_date, end=self.end_date)
        st.write()
        # Calculate financial metrics
        # Daily return (of close price)
        self.daily_return = self.stock_price['Close'].pct_change()

        # Volatility (of close price)
        self.daily_volatility = np.std(self.daily_return)

    def run_simulation(self):

        # Run the simulation
        np.random.seed(self.seed)
        self.simulation_df = pd.DataFrame()  # Reset

        for i in range(self.n_simulation):

            # The list to store the next stock price
            next_price = []

            # Create the next stock price
            last_price = self.stock_price['Close'][-1]

            for j in range(self.time_horizon):
                # Generate the random percentage change around the mean (0) and std (daily_volatility)
                future_return = np.random.normal(0, self.daily_volatility)

                # Generate the random future price
                future_price = last_price * (1 + future_return)

                # Save the price and go next
                next_price.append(future_price)
                last_price = future_price

            # Store the result of the simulation
            next_price_df = pd.Series(next_price).rename('sim' + str(i))
            self.simulation_df = pd.concat(
                [self.simulation_df, next_price_df], axis=1)

    def plot_simulation_price(self):

        # Plot the simulation stock price in the future
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10, forward=True)

        plt.plot(self.simulation_df)
        plt.title('Monte Carlo simulation for ' + self.ticker +
                  ' stock price in next ' + str(self.time_horizon) + ' days')
        plt.xlabel('Day')
        plt.ylabel('Price')

        plt.axhline(y=self.stock_price['Close'][-1], color='red')
        plt.legend(['Current stock price is: ' +
                   str(np.round(self.stock_price['Close'][-1], 2))])
        ax.get_legend().legendHandles[0].set_color('red')
        st.pyplot(fig)

    def plot_simulation_hist(self):

        # Get the ending price of the 200th day
        ending_price = self.simulation_df.iloc[-1:, :].values[0, ]

        # Plot using histogram
        fig, ax = plt.subplots()
        plt.hist(ending_price, bins=50)
        plt.axvline(x=self.stock_price['Close'][-1], color='red')
        plt.legend(['Current stock price is: ' +
                   str(np.round(self.stock_price['Close'][-1], 2))])
        ax.get_legend().legendHandles[0].set_color('red')
        st.pyplot(fig)

    def value_at_risk(self):
        # Price at 95% confidence interval
        future_price_95ci = np.percentile(
            self.simulation_df.iloc[-1:, :].values[0, ], 5)

        # Value at Risk
        VaR = self.stock_price['Close'][-1] - future_price_95ci
        print('VaR at 95% confidence interval is: ' +
              str(np.round(VaR, 2)) + ' USD')

# Function that has all the functionality for the Monte Carlo tab
def monte_carlo():
    # Add Monte Carlo tab title
    st.title("Monte Carlo simulation")

    # Create two columns to add the number of days and number of simulation selectors
    select1, select2 = st.columns(2)
    with select1:
        # Add radio selector with the number of simulations to run (Make them horizontal for better display)
        n_simulations = st.radio(
            "Number of simulations:",
            ["200", "500", "1000"],
            horizontal=True,
        )
    with select2:
        # Add radio selector with the number of days from today (Make them horizontal for better display)
        n_days = st.radio(
            "Number of days from today",
            ["30", "60", "90"],
            horizontal=True,
        )

    # Initiate the MonteCarlo Object with the needed values (ticker name, start date, end date, time horizon and number of simulations)
    mc_sim = MonteCarlo(ticker,
                        start_date=(date.today() - timedelta(int(n_days))), end_date=date.today(),
                        time_horizon=int(n_days), n_simulation=int(n_simulations), seed=123)

    # Run the simulation to get the results
    mc_sim.run_simulation()

    # Plot the simulation results into the dashboard
    mc_sim.plot_simulation_price()

# ==============================================================================
# Custom analysis
# ==============================================================================


def custom_analysis():
    # Add dashboard title and description
    st.title("Custom analysis")

    st.header('Dividends & splits')
    # Create two columns (dividends and splits)
    dividends, splits = st.columns(2)
    # Inside dividends column add the dividends
    with dividends:
        st.dataframe(yf.Ticker(ticker).dividends)
    # Inside the spits add the splits
    with splits:
        st.dataframe(yf.Ticker(ticker).splits)

    st.header('News')
    # Get the news information
    news = yf.Ticker(ticker).news

    # Go through all the news articles
    for newArticle in news:
        # Print the news article title and make it a link to go to the actual page
        st.subheader(f"[{newArticle['title']}]({newArticle['link']})")
        # Create the columns for the thumbnail (image preview) and the content itself.
        # The first column will have 1/4 of the width, and the second will have 3/4 of the width
        thumbnail, content = st.columns([1, 3])
        with thumbnail:
            # If the thumbnail exists (it might not exist in some news)
            if 'thumbnail' in newArticle:
                st.image(
                    # Get the image URL from the news data
                    newArticle['thumbnail']['resolutions'][0]['url'],
                    width=200,  # Manually Adjust the width of the image
                    use_column_width=True,
                )
        with content:
            # Write the publisher of the new article
            st.write(f"Publisher **{newArticle['publisher']}**")
            # Write the published date and format it nicely
            st.write(
                f"Published date **{datetime.fromtimestamp(newArticle['providerPublishTime']).strftime('%d %B, %Y')}**")
            # If there are any related tickers then display them nicely (joined by ,)
            if len(newArticle['relatedTickers']) > 1:
                st.write(
                    f"Related tickers\n\n **{', '.join(newArticle['relatedTickers'])}**"
                )

# ==============================================================================
# Main body
# ==============================================================================


def run():

    st.set_page_config(layout="wide")

    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    ticker_list = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

    # Add selection box for the tickers we read from Wikipedia
    global ticker
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list)

    global select_tab
    # Add a selection box for each tab
    select_tab = st.sidebar.selectbox(
        "Select tab", ['Summary', 'Chart', 'Financials', 'Monte Carlo', 'Custom analysis'])

    # Call the update tab to display the tab by default at the beginning
    update_tab()

    # Add update button which callbacs the update tab function to reload whatever tab we are on
    st.sidebar.button("Update", key="update_tab")

# Function that displays the tab that the user selects
def update_tab():
    # Show the selected tab
    if select_tab == 'Summary':
        # Run tab 1
        summary()
    elif select_tab == 'Chart':
        # Run tab 2
        chart()
    elif select_tab == 'Financials':
        # Run tab 3
        financial()
    elif select_tab == 'Monte Carlo':
        # Run tab 4
        monte_carlo()
    elif select_tab == 'Custom analysis':
        # Run tab 5
        custom_analysis()


if __name__ == "__main__":
    run()

###############################################################################
# END
###############################################################################
