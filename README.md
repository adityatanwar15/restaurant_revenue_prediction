# restaurant_revenue_prediction
Revenue prediction of a test set of 100000 restaurants with only a 137 restaurants train dataset.
import pandas as pd
import numpy as np
import plotly.graph_objects as go

data = pd.read_excel('eurusd-forex-data.xlsx')

# Scalling the price to calculate much better
data['Open'] = data['Open'] * 1000
data['High'] = data['High'] * 1000
data['Low'] = data['Low'] * 1000
data['Close'] = data['Close'] * 1000
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

data.set_index('Timestamp', inplace=True)

#CMF
def calculate_cmf(data, period):
    data['Money Flow Multiplier'] = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    data['Money Flow Volume'] = data['Money Flow Multiplier'] * data['Volume']
    cmf = data['Money Flow Volume'].rolling(window=period).sum() / data['Volume'].rolling(window=period).sum()
    return cmf

# Calculate RSI
def calculate_rsi(data, period=14):
    close_prices = data['Close']
    price_changes = close_prices.diff()
    momentum_up = price_changes.where(price_changes > 0, 0)
    momentum_down = -price_changes.where(price_changes < 0, 0)
    avg_momentum_up = momentum_up.rolling(window=period).mean()
    avg_momentum_down = momentum_down.rolling(window=period).mean()
    rsi = 100 * (avg_momentum_up / (avg_momentum_up + avg_momentum_down))
    return rsi

def calculate_bollinger_bands(data, period=14, num_std_dev=1.5):
    data['Middle Band'] = data['Close'].rolling(window=period).mean()
    data['Upper Band'] = data['Middle Band'] + (data['Close'].rolling(window=period).std() * num_std_dev)
    data['Lower Band'] =data['Middle Band'] - (data['Close'].rolling(window=period).std() * num_std_dev)

data['CMF_10'] = calculate_cmf(data, 10)
data['CMF_20'] = calculate_cmf(data, 20)
data['CMF_50'] = calculate_cmf(data, 50)
data['RSI'] = calculate_rsi(data)
calculate_bollinger_bands(data)

"""Signal generation, this is an indicator based strategy where I have used 3 different indicator to predict oversold and overbought of the contract 
and utilized a cap on holding period of the contact  and averaging on the dynamic target and stoploss which is calculated by the SMA of candle lenght 
and rolling 14 days standard deviation, everytime an the same order comes by the signal. It can be improve by giving a score to the signal generated."""
data['Buy Signal'] = ((data['CMF_10'] < -0.30) | (data['CMF_20'] < -0.22) | (data['CMF_50'] < -0.12)) & (data['RSI'] < 35) & (data['Close'] <= data['Lower Band'])
data['Sell Signal'] = ((data['CMF_10'] > 0.30) | (data['CMF_20'] > 0.20) | (data['CMF_50'] > 0.1)) & (data['RSI'] > 65) & (data['Close'] >= data['Upper Band'])

data['Candle_Length'] = ((data['High'] - data['Low']) + (data['Close'] - data['Open'])) / 2
data['Avg_Candle_Length'] = data['Candle_Length'].rolling(window=14).mean()
data['14 Day SD'] = data['Close'].rolling(window=14).std()
data.dropna(inplace=True)

def backtest(data):
    position = 0  # 1 : long, -1 : short
    entry_prices = []  
    entry_dates = []  
    holding_days = 0  
    trade_log = []

    for index, row in data.iterrows():
        if row['Buy Signal'] and position >= 0:  # Buy signal
            if position == 0:  
                position = 1
                entry_prices.append(row['Close'])
                entry_dates.append(row.name)
                holding_days = 0  # Reset the holding days
            else:  
                entry_prices.append(row['Close'])
                holding_days += 1  

        elif row['Sell Signal'] and position <= 0: # Sell signal 
            if position == 0:  
                position = -1
                entry_prices.append(row['Close'])
                entry_dates.append(row.name)
                holding_days = 0  # Reset holding days
            else: 
                entry_prices.append(row['Close'])
                holding_days += 1  

        if position == 1:  # buy position
            holding_days += 1
            avg_entry_price = np.mean(entry_prices)
            stop_loss = avg_entry_price - (data['Avg_Candle_Length'].loc[index] + data['14 Day SD'].loc[index])
            target = avg_entry_price + 4.5 * data['Avg_Candle_Length'].loc[index] + data['14 Day SD'].loc[index]

            if row['Close'] <= stop_loss or row['Close'] >= target or holding_days >= 12:
                exit_price = row['Close']
                outcome = 'Win' if exit_price >= target else 'Loss' if row['Close'] <= stop_loss else 'Holding Period Hit'
                return_value = (exit_price - avg_entry_price) / avg_entry_price

                for entry_price in entry_prices:
                    trade_log.append({
                        'Entry Date': entry_dates[0],  # First entry date
                        'Entry Price': entry_price,
                        'Exit Date': row.name,
                        'Exit Price': exit_price,
                        'Stop Loss': stop_loss,
                        'Target': target,
                        'Outcome': outcome,
                        'Trade Type': 'Buy',
                        'Return': return_value
                    })

                position = 0  
                entry_prices.clear()  
                entry_dates.clear()  
                holding_days = 0  

        elif position == -1:  # sell position
            holding_days += 1
            avg_entry_price = np.mean(entry_prices)
            stop_loss = avg_entry_price + (data['Avg_Candle_Length'].loc[index] + data['14 Day SD'].loc[index])
            target = avg_entry_price - 5 * data['Avg_Candle_Length'].loc[index] - data['14 Day SD'].loc[index]

            if row['Close'] >= stop_loss or row['Close'] <= target or holding_days >= 15:
                exit_price = row['Close']
                outcome = 'Win' if exit_price <= target else 'Loss' if row['Close'] >= stop_loss else 'Holding Period Hit'
                return_value = (avg_entry_price - exit_price) / avg_entry_price

                for entry_price in entry_prices:
                    trade_log.append({
                        'Entry Date': entry_dates[0],  
                        'Entry Price': entry_price,
                        'Exit Date': row.name,
                        'Exit Price': exit_price,
                        'Stop Loss': stop_loss,
                        'Target': target,
                        'Outcome': outcome,
                        'Trade Type': 'Sell',
                        'Return': return_value
                    })

                position = 0  
                entry_prices.clear()  
                entry_dates.clear()  
                holding_days = 0  

    return trade_log

trade_log = backtest(data)
trade_log_df = pd.DataFrame(trade_log)
trade_log_df['Cumulative Return'] = (1 + trade_log_df['Return']).cumprod()
cumulative_returns = trade_log_df.groupby('Exit Date')['Cumulative Return'].last().reset_index()
fig_returns = go.Figure()
fig_returns.add_trace(go.Scatter(x=cumulative_returns['Exit Date'],
                                   y=cumulative_returns['Cumulative Return'],
                                   mode='lines+markers',
                                   name='Cumulative Return',
                                   line=dict(color='blue')))

fig_returns.update_layout(
    title='Cumulative Returns Over Time',
    xaxis_title='Date',
    yaxis_title='Cumulative Return',
    xaxis_rangeslider_visible=False
)
fig_returns.show()
data['Date'] = data.index
data.set_index('Date', inplace=True)
buy_signals = data[data['Buy Signal']]
sell_signals = data[data['Sell Signal']]
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index,
                              open=data['Open'],
                              high=data['High'],
                              low=data['Low'],
                              close=data['Close'],
                              name='Candlestick'))

fig.add_trace(go.Scatter(x=buy_signals.index,
                         y=buy_signals['Close'],
                         mode='markers',
                         marker=dict(symbol='triangle-up', size=10, color='green'),
                         name='Buy Signal'))

fig.add_trace(go.Scatter(x=sell_signals.index,
                         y=sell_signals['Close'],
                         mode='markers',
                         marker=dict(symbol='triangle-down', size=10, color='red'),
                         name='Sell Signal'))

fig.add_trace(go.Scatter(x=data.index,
                         y=data['Upper Band'],
                         mode='lines',
                         line=dict(color='blue', dash='dash'),
                         name='Upper Band'))

fig.add_trace(go.Scatter(x=data.index,
                         y=data['Middle Band'],
                         mode='lines',
                         line=dict(color='orange', dash='dash'),
                         name='Middle Band'))

fig.add_trace(go.Scatter(x=data.index,
                         y=data['Lower Band'],
                         mode='lines',
                         line=dict(color='blue', dash='dash'),
                         name='Lower Band'))

fig.add_trace(go.Scatter(x=data.index,
                         y=data['CMF_10'],
                         mode='lines',
                         line=dict(color='purple'),
                         name='CMF (10 Periods)',
                         yaxis='y2'))

fig.add_trace(go.Scatter(x=data.index,
                         y=data['CMF_20'],
                         mode='lines',
                         line=dict(color='red'),
                         name='CMF (20 Periods)',
                         yaxis='y2'))

fig.add_trace(go.Scatter(x=data.index,
                         y=data['CMF_50'],
                         mode='lines',
                         line=dict(color='green'),
                         name='CMF (50 Periods)',
                         yaxis='y2'))

fig.add_trace(go.Scatter(x=data.index,
                         y=data['RSI'],
                         mode='lines',
                         line=dict(color='magenta'),
                         name='RSI',
                         yaxis='y2'))

fig.update_layout(
    title='Trading Strategy with Bollinger Bands and Indicators',
    xaxis_title='Date',
    yaxis_title='Price',
    yaxis2=dict(title='Indicators', overlaying='y', side='right', showgrid=False),
    xaxis_rangeslider_visible=False
)
fig.show()
def performance_metrics(trade_log_df):
    trade_log_df['Return'] = np.where(
        trade_log_df['Trade Type'] == 'Buy',
        (trade_log_df['Exit Price'] - trade_log_df['Entry Price']) / trade_log_df['Entry Price'],
        (trade_log_df['Entry Price'] - trade_log_df['Exit Price']) / trade_log_df['Entry Price']
    )
    total_return = trade_log_df['Return'].sum() * 100  
    n_years = (trade_log_df['Exit Date'].iloc[-1] - trade_log_df['Entry Date'].iloc[0]).days / 365.25 if not trade_log_df.empty else 0
    final_value = (1 + trade_log_df['Return']).prod() 
    initial_value = 1  
    cagr = (final_value / initial_value) ** (1 / n_years) - 1 if n_years > 0 else 0
    win_ratio = len(trade_log_df[trade_log_df['Return'] > 0]) / len(trade_log_df) if len(trade_log_df) > 0 else 0
    average_return = trade_log_df['Return'].mean()
    risk_free_rate = 0.0  
    excess_return = average_return - risk_free_rate
    sharpe_ratio = excess_return / trade_log_df['Return'].std() * np.sqrt(252) if trade_log_df['Return'].std() > 0 else 0
    downside_returns = trade_log_df['Return'][trade_log_df['Return'] < 0]
    sortino_ratio = excess_return / downside_returns.std() * np.sqrt(252) if downside_returns.std() > 0 else 0
    cumulative_returns = (1 + trade_log_df['Return']).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (peak - cumulative_returns) / peak
    max_drawdown = drawdown.max() * 100 
    calmer_ratio = (average_return / max_drawdown) * 100 if max_drawdown != 0 else 0
    return {
        'Total Return (%)': total_return,
        'CAGR (%)': cagr * 100, 
        'Win Ratio': win_ratio,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Maximum Drawdown (%)': max_drawdown,
        'Calmer Ratio': calmer_ratio,
    }
metrics = performance_metrics(trade_log_df)
print(metrics)
print(trade_log_df)


