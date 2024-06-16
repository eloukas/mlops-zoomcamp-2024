---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.0
  nbformat: 4
  nbformat_minor: 5
---

<div class="cell code" execution_count="1">

``` python
import os
import sys

if os.name == "nt":  # For Windows
    !pip freeze | findstr scikit-learn
else:  # For Linux/macOS
    !pip freeze | grep scikit-learn
```

<div class="output stream stdout">

    scikit-learn==1.5.0

</div>

</div>

<div class="cell code" execution_count="2">

``` python
!python -V
```

<div class="output stream stdout">

    Python 3.9.0

</div>

</div>

<div class="cell code" execution_count="6">

``` python
import pickle
import pandas as pd
```

</div>

<div class="cell code" execution_count="7">

``` python
with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)
```

<div class="output stream stderr">

    c:\Users\kaslou\anaconda3\envs\mlops-zoomcamp\lib\site-packages\sklearn\base.py:376: InconsistentVersionWarning: Trying to unpickle estimator DictVectorizer from version 1.5.0 when using version 1.4.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
    https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
      warnings.warn(
    c:\Users\kaslou\anaconda3\envs\mlops-zoomcamp\lib\site-packages\sklearn\base.py:376: InconsistentVersionWarning: Trying to unpickle estimator LinearRegression from version 1.5.0 when using version 1.4.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
    https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations
      warnings.warn(

</div>

</div>

<div class="cell code" execution_count="8">

``` python
categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df
```

</div>

<div class="cell code" execution_count="9">

``` python
# Read Yellow Taxi Trip Records  March 2023 data
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

df = read_data(url)
```

</div>

<div class="cell code" execution_count="10">

``` python
dicts = df[categorical].to_dict(orient="records")
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
```

</div>

<div class="cell markdown">

## Q1. Notebook (Question 1)

We'll start with the same notebook we ended up with in homework 1. We
cleaned it a little bit and kept only the scoring part. You can find the
initial notebook here.

Run this notebook for the March 2023 data.

What's the standard deviation of the predicted duration for this
dataset?

-   1.24
-   6.24
-   12.28
-   18.28

</div>

<div class="cell code" execution_count="12">

``` python
# Find the standard deviation of the predicted duration
predicted_duration_std = y_pred.std()

print(
    f" The standard deviation of the predicted duration is {predicted_duration_std:.2f} minutes"
)

# The standard deviation of the predicted duration is 6.25 minutes
# So, answer is option B, 6.24 (closest to 6.25)
```

<div class="output stream stdout">

     The standard deviation of the predicted duration is 6.25 minutes

</div>

</div>

<div class="cell markdown">

## Q2. Preparing the output

Like in the course videos, we want to prepare the dataframe with the
output.

First, let's create an artificial ride_id column:

`df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')`

Next, write the ride id and the predictions to a dataframe with results.

Save it as parquet:

``` python
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
```

What's the size of the output file?

-   36M
-   46M
-   56M
-   66M

Note: Make sure you use the snippet above for saving the file. It should
contain only these two columns. For this question, don't change the
dtypes of the columns and use pyarrow, not fastparquet.

</div>

<div class="cell code" execution_count="13">

``` python
# First, let's create an artificial ride_id column

year = 2023
month = 3

df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

# Next, write the ride id and the predictions to a dataframe with results.

results = pd.DataFrame({"ride_id": df["ride_id"], "predicted_duration": y_pred})

print(results.head())
```

<div class="output execute_result" execution_count="13">

         ride_id  predicted_duration
    0  2023/03_0           16.245906
    1  2023/03_1           26.134796
    2  2023/03_2           11.884264
    3  2023/03_3           11.997720
    4  2023/03_4           10.234486

</div>

</div>

<div class="cell code" execution_count="14">

``` python
# Save it as parquet file

output_file = f"{year:04d}-{month:02d}-predictions.parquet"

results.to_parquet(output_file, engine="pyarrow", compression=None, index=False)

print(f"Predictions saved to {output_file}")
```

<div class="output stream stdout">

    Predictions saved to 2023-03-predictions.parquet

</div>

</div>

<div class="cell code" execution_count="15">

``` python
# What is the size of the saved file (in MB)?

file_size = os.path.getsize(output_file) / (1024 * 1024)
print(f"Size of the saved file is {file_size:.2f} MB")

# Size of the saved file is 65.46 MB
# So, answer is option D, 66MB (closest to 65.46)
```

<div class="output stream stdout">

    Size of the saved file is 65.46 MB

</div>

</div>

<div class="cell markdown">

## Q3. Creating the scoring script

Now let's turn the notebook into a script.

Which command you need to execute for that?

</div>

<div class="cell code" execution_count="16">

``` python
# Now, let's just turn the notebook into a script using jupyter nbconvert

!jupyter nbconvert --to script --output-dir . starter.ipynb

# List the files in the current directory (Windows or Linux/macOS)
if os.name == "nt":  # For Windows
    !dir
else:
    !ls -l

# The script is saved as starter.py
```

<div class="output stream stderr">

    [NbConvertApp] Converting notebook starter.ipynb to script
    [NbConvertApp] Writing 3572 bytes to starter.py

</div>

<div class="output stream stdout">

     Volume in drive C has no label.
     Volume Serial Number is E2F7-236A

     Directory of c:\Users\kaslou\Desktop\code\mlops-zoomcamp-2024\week4

    16-Jun-24  11:36 AM    <DIR>          .
    16-Jun-24  11:36 AM    <DIR>          ..
    16-Jun-24  11:33 AM        68,641,880 2023-03-predictions.parquet
    16-Jun-24  11:24 AM            17,376 model.bin
    16-Jun-24  11:34 AM            10,332 starter.ipynb
    16-Jun-24  11:36 AM             3,752 starter.py
                   4 File(s)     68,673,340 bytes
                   2 Dir(s)  162,013,110,272 bytes free

</div>

</div>

<div class="cell markdown">

## Q4. Virtual environment

Now let's put everything into a virtual environment. We'll use pipenv
for that.

Install all the required libraries. Pay attention to the Scikit-Learn
version: it should be the same as in the starter notebook.

After installing the libraries, pipenv creates two files: `Pipfile` and
`Pipfile.lock`. The Pipfile.lock file keeps the hashes of the
dependencies we use for the virtual env.

Question: What's the first hash for the Scikit-Learn dependency?

</div>

<div class="cell code" execution_count="17">

``` python
# The first hash is ""sha256:057b991ac64b3e75c9c04b5f9395eaf19a6179244c089afdebaad98264bff37c"
```

</div>

<div class="cell markdown">

## Q5. Parametrize the script

Let's now make the script configurable via CLI. We'll create two
parameters: year and month.

Run the script for April 2023.

What's the mean predicted duration?

-   7.29
-   14.29
-   21.29
-   28.29

Hint: just add a print statement to your script.

</div>

<div class="cell code" execution_count="18">

``` python
# For that, I switch to the starter.py script. Refer to it to see the changes.


# The mean predicted duration for 2023-04 is 14.29 minutes
# Predictions saved to 2023-04-predictions.parquet

# So, the answer is option B, 14.29


```

</div>

<div class="cell markdown">

## Q6. Docker container

Finally, we'll package the script in the docker container. For that,
you'll need to use a base image that we prepared.

This is what the content of this image is:

    FROM python:3.10.13-slim

    WORKDIR /app
    COPY [ "model2.bin", "model.bin" ]

Note: you don't need to run it. We have already done it.

It is pushed it to
[`agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim`](https://hub.docker.com/layers/agrigorev/zoomcamp-model/mlops-2024-3.10.13-slim/images/sha256-f54535b73a8c3ef91967d5588de57d4e251b22addcbbfb6e71304a91c1c7027f?context=repo),
which you need to use as your base image.

That is, your Dockerfile should start with:

``` docker
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# do stuff here
```

This image already has a pickle file with a dictionary vectorizer and a
model. You will need to use them.

Important: don't copy the model to the docker image. You will need to
use the pickle file already in the image.

Now run the script with docker. What's the mean predicted duration for
May 2023?

-   0.19
-   7.24
-   14.24
-   21.19

## Bonus: upload the result to the cloud (Not graded)

Just printing the mean duration inside the docker image doesn't seem
very practical. Typically, after creating the output file, we upload it
to the cloud storage.

Modify your code to upload the parquet file to S3/GCS/etc.

## Publishing the image to dockerhub

This is how we published the image to Docker hub:

``` bash
docker build -t mlops-zoomcamp-model:2024-3.10.13-slim .
docker tag mlops-zoomcamp-model:2024-3.10.13-slim agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

docker login --username USERNAME
docker push agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
```

</div>

<div class="cell code" execution_count="3">

``` python
# I created the inference.dockerfile file.
# Then, I built the docker image using the following command:
# docker build -t ride-duration-pred-service:v1 -f week4/inference.dockerfile .

# I ran the docker container using the following command:
# docker run -it --rm ride-duration-pred-service:v1 --year 2023 --month 5

# (mlops-zoomcamp-2024-ZOLEji97) C:\Users\kaslou\Desktop\code\mlops-zoomcamp-2024>docker run -it --rm ride-duration-pred-service:v1 --year 2023 --month 5
# The mean predicted duration for 2023-05 is 0.19 minutes
# Predictions saved to 2023-05-predictions.parquet


```

</div>

<div class="cell code">

``` python
```

</div>
