import os

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

DATA_FOLDER = r"C:\Users\kaslou\Desktop\code\mlops-zoomcamp-2024\week1\hw1"

# Q1: Read the data for January. How many columns are there?

# Load the parquet file
january_parquet_file_path = "yellow_tripdata_2023-01.parquet"
january_parquet_file_path = os.path.join(DATA_FOLDER, january_parquet_file_path)
df = pd.read_parquet(january_parquet_file_path)


# Get the number of columns and their names
num_columns = len(df.columns)
column_names = df.columns.tolist()
num_records = len(df)

print(f"Number of columns: {num_columns}")
print(f"Column names: {column_names}")
# Number of columns: 19
# Column names: ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'airport_fee']

# Q2: Now let's compute the duration variable. It should contain the duration of a ride in minutes.
# What's the standard deviation of the trips duration in January?

# Convert the pickup and dropoff datetime columns to datetime objects
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

# Compute the duration in minutes
df["duration"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60

# Compute the standard deviation of the duration
std_dev_duration = df["duration"].std()

print(f"Standard deviation of trip duration in January: {std_dev_duration}")
# Standard deviation of trip duration in January: 42.59435124195458

# Q3. Dropping outliers
# Next, we need to check the distribution of the duration variable. There are some outliers. Let's remove them and keep only the records where the duration was between 1 and 60 minutes (inclusive).
# What fraction of the records left after you dropped the outliers?

# Drop the outliers
df = df[df["duration"].between(1, 60)]

# Compute the fraction of records left
fraction_left = len(df) / num_records
print(f"Fraction of records left after dropping outliers: {fraction_left}")
# Fraction of records left after dropping outliers: 0.9812202822125979

# Q4. One-hot encoding
# Let's apply one-hot encoding to the pickup and dropoff location IDs. We'll use only these two features for our model.
# Turn the dataframe into a list of dictionaries (remember to re-cast the ids to strings - otherwise it will label encode them)
# Fit a dictionary vectorizer
# Get a feature matrix from it
# What's the dimensionality of this matrix (number of columns)?


# Convert the dataframe into a list of dictionaries
# Extract relevant columns and convert IDs to strings
df["PULocationID"] = df["PULocationID"].astype(str)
df["DOLocationID"] = df["DOLocationID"].astype(str)
relevant_columns = df[["PULocationID", "DOLocationID", "duration"]]

# Convert dataframe to list of dictionaries
data_dicts = relevant_columns.to_dict(orient="records")

# Fit a dictionary vectorizer
dv = DictVectorizer(sparse=True)
feature_matrix = dv.fit_transform(data_dicts)

# Get the dimensionality of the matrix
num_columns = feature_matrix.shape[1]

print(f"The dimensionality of the feature matrix is: {num_columns}")
# The dimensionality of the feature matrix is: 515

# Q5. Training a model
# Now let's use the feature matrix from the previous step to train a model.
# Train a plain linear regression model with default parameters
# Calculate the RMSE of the model on the training data
# What's the RMSE on train?

categorical = ["PULocationID", "DOLocationID"]

df[categorical] = df[categorical].astype("str")


february_parquet_file_path = "yellow_tripdata_2023-02.parquet"
# Load the validation data
february_parquet_file_path = os.path.join(DATA_FOLDER, february_parquet_file_path)
df2 = pd.read_parquet(february_parquet_file_path)

df2["duration"] = df2.tpep_dropoff_datetime - df2.tpep_pickup_datetime
df2.duration = df2.duration.apply(lambda td: td.total_seconds() / 60)
df2 = df2[(df2.duration >= 1) & (df2.duration <= 60)]

df[categorical] = df[categorical].astype("str")
df2[categorical] = df2[categorical].astype("str")

train_dict = df[categorical].to_dict(orient="records")
val_dict = df2[categorical].to_dict(orient="records")


target = "duration"
dv = DictVectorizer()
x_train = dv.fit_transform(train_dict)
y_train = df[target].values

target = "duration"
x_val = dv.transform(val_dict)
y_val = df2[target].values


lr = LinearRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_train)

y_val_predict = lr.predict(x_val)
print(f"RMSE on train: {mean_squared_error(y_train, y_predict, squared=False)}")
# RMSE on train: 7.649261931416412

# Q6. Evaluating the model
# Now let's apply this model to the validation dataset (February 2023).

# What's the RMSE on validation?
print(f"RMSE on val: {mean_squared_error(y_val, y_val_predict, squared=False)}")
