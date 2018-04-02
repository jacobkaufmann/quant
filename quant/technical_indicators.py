from sklearn import preprocessing

# Use of this library requires 'Volume' and 'Adj Close' columns


# Calculate average volume over a specified interval of periods
def calculate_avg_vol_for_interval(security, periods):
    security['{}d_Avg_Vol'.format(periods)] = security['Volume'].rolling(window=periods).mean()


# Calculate simple moving average over a specified interval of periods
def calculate_sma_for_interval(security, periods):
    security['{}d_SMA'.format(periods)] = security['Adj Close'].rolling(window=periods).mean()


# Calculate exponential moving average over a specified interval of periods
def calculate_ema_for_interval(security, periods):
    security['{}d_EMA'.format(periods)] = security['Adj Close'].ewm(span=periods).mean()


# Calculate low over specified interval of periods for specified column
def calculate_low_for_interval(security, col, periods):
    security['{}_{}d_Low'.format(col, periods)] = security[col].rolling(window=periods).min()


# Calculate high over specified interval of periods for specified column
def calculate_high_for_interval(security, col, periods):
    security['{}_{}d_High'.format(col, periods)] = security[col].rolling(window=periods).max()


# Calculate per day price changes with columns for gain and loss
def calculate_per_day_price_change(security):
    security['Price_Change'] = security['Adj Close'].diff(periods=1)
    security['Gain'] = [x if x > 0 else 0 for x in security['Price_Change']]
    security['Loss'] = [-x if x < 0 else 0 for x in security['Price_Change']]


# Calculate average gain over specified interval of periods
def calculate_avg_gain_for_interval(security, periods):
    security['{}d_Avg_Gain'.format(periods)] = security['Gain'].rolling(window=periods).sum() / periods


# Calculate average loss over specified interval of periods
def calculate_avg_loss_for_interval(security, periods):
    security['{}d_Avg_Loss'.format(periods)] = security['Loss'].rolling(window=periods).sum() / periods


# Calculate RSI for specified interval of periods
def calculate_rsi(security, periods):
    calculate_avg_gain_for_interval(security, periods)
    calculate_avg_loss_for_interval(security, periods)

    security['{}d_RSI'.format(periods)] = 100 - \
        (100 / (1 + (security['{}d_Avg_Gain'.format(periods)] / security['{}d_Avg_Loss'.format(periods)])))
    security.drop(['Price_Change', 'Gain', 'Loss', '{}d_Avg_Gain'.format(periods), '{}d_Avg_Loss'.format(periods)],
                  axis=1, inplace=True)


# Calculate MACD, MACD Signal Line, and MACD Histogram for specified short, long, and signal parameters
def calculate_macd(security, short, long, signal):
    calculate_ema_for_interval(security, short)
    calculate_ema_for_interval(security, long)

    security['MACD_{}_{}_{}'.format(short, long, signal)] = security['{}d_EMA'.format(short)] - security[
        '{}d_EMA'.format(long)]

    security['MACD_{}_{}_{}_Signal'.format(short, long, signal)] = security[
        'MACD_{}_{}_{}'.format(short, long, signal)].ewm(span=signal).mean()

    security['MACD_{}_{}_{}_Hist'.format(short, long, signal)] = \
        security['MACD_{}_{}_{}'.format(short, long, signal)] - \
        security['MACD_{}_{}_{}_Signal'.format(short, long, signal)]


# Calculate average rate of change for a particular column over a specified interval of periods
def calculate_avg_roc_for_interval(security, column, periods):
    security['{}_{}d_Avg_ROC'.format(column, periods)] = (security[column] - security[column].shift(periods)) / periods


# Standardize a column
def scale_col(security, column):
    security[column] = preprocessing.scale(security[column])


# Calculate ratio between two columns
def calculate_ratio(security, col1, col2):
    try:
        security['{}_v_{}'.format(col1, col2)] = security[col1] / security[col2]
    except ValueError:
        print('Ratio failed to be computed')


# Calculate Accumulation Distribution Line (ADL)
def calculate_adl(security):
    mfm = ((security['Close'] - security['Low']) - (security['High'] - security['Close'])) / (security['High'] - security['Low'])
    mfv = mfm * security['Volume']
    security['MFV'] = mfv
    security['ADL'] = security['MFV'].cumsum()


# Calculate Full Stochastics
def calculate_full_stochs(security, periods, s1, s2):
    security['%K'] = (security['Close'] - security['Low'].rolling(window=periods).min()) / \
                   (security['High'].rolling(window=periods).max() - security['Low'].rolling(window=periods).min()) * 100
    security['%K_Full'] = security['%K'].rolling(window=s1).mean()
    security['%D_Full'] = security['%K_Full'].rolling(window=s2).mean()
    security.drop(['%K'], axis=1, inplace=True)


# Calculate Chaikin Oscillator
def calculate_chaikin_osc(security):
    security['Chaikin'] = security['ADL'].ewm(span=3).mean() - security['ADL'].ewm(span=10).mean()

