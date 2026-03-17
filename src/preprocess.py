import pandas as pd

def preprocess_data(df):
    # Create datetime from date + time
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    else:
        raise Exception("No valid datetime columns found")

    # Feature engineering
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    df['is_peak'] = df['hour'].apply(lambda x: 1 if 7<=x<=10 or 17<=x<=20 else 0)

    df = df.fillna(0)
    df = pd.get_dummies(df, drop_first=True)
    
    return df