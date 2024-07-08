#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import numpy as np
import sklearn

def read_data(filename, categorical):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def prepare_data(df, categorical, year, month):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df


def main(year, month):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)

    df = prepare_data(df, categorical, year, month)
    dicts = df[categorical].to_dict(orient='records')
    df[categorical].to_dict(orient='records')

    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']

test_data = [[np.nan, np.nan, pd.Timestamp('2023-01-01 01:01:00'), pd.Timestamp('2023-01-01 01:10:00'), '2023/02_0'], [1.0, 1.0, pd.Timestamp('2023-01-01 01:02:00'), pd.Timestamp('2023-01-01 01:10:00'), '2023/02_1'], [1.0, np.nan, pd.Timestamp('2023-01-01 01:02:00'), pd.Timestamp('2023-01-01 01:02:59'), '2023/02_2'], [3.0, 4.0, pd.Timestamp('2023-01-01 01:02:00'), pd.Timestamp('2023-01-01 02:02:01'), '2023/02_3']]
def test():
    categorical = ['PULocationID', 'DOLocationID']
    df = pd.DataFrame(data, columns=columns)
    df2 = prepare_data(df, categorical, 2023, 3)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    df[categorical].to_dict(orient='records')

    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())
    print('predicted mean duration:', y_pred.sum())
    # df2.to_parquet(
    #     "a.pkl",
    #     engine='pyarrow',
    #     compression=None,
    #     index=False
    # )
    # ret_val = df2.values.tolist()
    # print(ret_val)
    # print(test_data)
    # assert ret_val == test_data
    # print()

test()

# if __name__ == "__main__":
#     year = int(sys.argv[1])
#     month = int(sys.argv[2])
#     main(year, month)
