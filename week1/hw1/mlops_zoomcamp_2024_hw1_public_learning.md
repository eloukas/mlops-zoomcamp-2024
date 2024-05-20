
# Homework 1

This script performs several tasks on New York City taxi trip data to explore, preprocess, and model the data using linear regression.

## Step 1: Data Loading

Read the data for January and get the number of columns and their names.

```python
import os
import pandas as pd

DATA_FOLDER = r"C:\Users\kaslou\Desktop\code\mlops-zoomcamp-2024\week1\hw1"
january_parquet_file_path = "yellow_tripdata_2023-01.parquet"
january_parquet_file_path = os.path.join(DATA_FOLDER, january_parquet_file_path)
df = pd.read_parquet(january_parquet_file_path)

num_columns = len(df.columns)
column_names = df.columns.tolist()
num_records = len(df)

print(f"Number of columns: {num_columns}")
print(f"Column names: {column_names}")
```

**Result**:
- Number of columns: 19
- Column names: ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'airport_fee']

## Step 2: Compute Trip Duration

Calculate the duration of each trip in minutes and determine the standard deviation of these durations.

```python
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

df["duration"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60

std_dev_duration = df["duration"].std()
print(f"Standard deviation of trip duration in January: {std_dev_duration}")
```

**Result**:
- Standard deviation of trip duration in January: 42.59435124195458

## Step 3: Dropping Outliers

Remove trips with durations less than 1 minute or greater than 60 minutes and calculate the fraction of records remaining.

```python
df = df[df["duration"].between(1, 60)]
fraction_left = len(df) / num_records
print(f"Fraction of records left after dropping outliers: {fraction_left}")
```

**Result**:
- Fraction of records left after dropping outliers: 0.9812202822125979

## Step 4: One-Hot Encoding

Apply one-hot encoding to the pickup and dropoff location IDs and create a feature matrix.

```python
from sklearn.feature_extraction import DictVectorizer

df["PULocationID"] = df["PULocationID"].astype(str)
df["DOLocationID"] = df["DOLocationID"].astype(str)
relevant_columns = df[["PULocationID", "DOLocationID", "duration"]]

data_dicts = relevant_columns.to_dict(orient="records")
dv = DictVectorizer(sparse=True)
feature_matrix = dv.fit_transform(data_dicts)

num_columns = feature_matrix.shape[1]
print(f"The dimensionality of the feature matrix is: {num_columns}")
```

**Result**:
- The dimensionality of the feature matrix is: 515

## Step 5: Training a Model

Train a linear regression model on the January data and calculate the RMSE on the training set.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

categorical = ["PULocationID", "DOLocationID"]
df[categorical] = df[categorical].astype("str")

train_dict = df[categorical].to_dict(orient="records")
target = "duration"
x_train = dv.fit_transform(train_dict)
y_train = df[target].values

lr = LinearRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_train)
print(f"RMSE on train: {mean_squared_error(y_train, y_predict, squared=False)}")
```

**Result**:
- RMSE on train: 7.649261931416412

## Step 6: Evaluating the Model

Evaluate the model on the February data and calculate the RMSE on the validation set.

```python
february_parquet_file_path = "yellow_tripdata_2023-02.parquet"
february_parquet_file_path = os.path.join(DATA_FOLDER, february_parquet_file_path)
df2 = pd.read_parquet(february_parquet_file_path)

df2["duration"] = df2.tpep_dropoff_datetime - df2.tpep_pickup_datetime
df2.duration = df2.duration.apply(lambda td: td.total_seconds() / 60)
df2 = df2[(df2.duration >= 1) & (df2.duration <= 60)]

df2[categorical] = df2[categorical].astype("str")
val_dict = df2[categorical].to_dict(orient="records")

x_val = dv.transform(val_dict)
y_val = df2[target].values

y_val_predict = lr.predict(x_val)
print(f"RMSE on val: {mean_squared_error(y_val, y_val_predict, squared=False)}")
```

**Result**:
- RMSE on val: 7.795657391627858
