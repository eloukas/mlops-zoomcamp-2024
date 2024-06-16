#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle

import pandas as pd


def read_data(filename):
    """Read the data from the specified file and perform necessary transformations.

    Args:
        filename : str : The path to the parquet file to read

    Returns:
        pd.DataFrame : The transformed data

    """
    categorical = ["PULocationID", "DOLocationID"]
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def main(year, month):
    """
    Load the model and predict the trip durations for the specified month and year.

    Args:
        year : int : The year of the data to predict
        month : int : The month of the data to predict

    """
    # Load the model
    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    # Read Yellow Taxi Trip Records for the specified month and year
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    df = read_data(url)

    # Prepare the data for prediction
    categorical = ["PULocationID", "DOLocationID"]
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Calculate the mean predicted duration
    mean_predicted_duration = y_pred.mean()
    print(
        f"The mean predicted duration for {year}-{month:02d} is {mean_predicted_duration:.2f} minutes"
    )

    # Create the ride_id column and prepare results dataframe
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")
    results = pd.DataFrame({"ride_id": df["ride_id"], "predicted_duration": y_pred})

    # Save the results to a parquet file
    output_file = f"{year:04d}-{month:02d}-predictions.parquet"
    results.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict taxi trip durations")
    parser.add_argument(
        "--year", type=int, required=True, help="Year of the data to predict"
    )
    parser.add_argument(
        "--month", type=int, required=True, help="Month of the data to predict"
    )
    args = parser.parse_args()
    main(args.year, args.month)

# Example usage
## python starter.py --year 2020 --month 1 (run this command in the terminal if you want to have predictions for January 2020)

# Example output
## The mean predicted duration for 2020-01 is 12.34 minutes
## Predictions saved to 2020-01-predictions.parquet
