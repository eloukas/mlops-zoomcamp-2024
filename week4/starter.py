#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import sys

if os.name == "nt":  # For Windows
    get_ipython().system('pip freeze | findstr scikit-learn')
else:  # For Linux/macOS
    get_ipython().system('pip freeze | grep scikit-learn')


# In[5]:


get_ipython().system('python -V')


# In[6]:


import pickle
import pandas as pd


# In[7]:


with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


# In[8]:


categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


# In[9]:


# Read Yellow Taxi Trip Records  March 2023 data
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

df = read_data(url)


# In[10]:


dicts = df[categorical].to_dict(orient="records")
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# ## Q1. Notebook (Question 1)
# 
# We'll start with the same notebook we ended up with in homework 1. We cleaned it a little bit and kept only the scoring part. You can find the initial notebook here.
# 
# Run this notebook for the March 2023 data.
# 
# What's the standard deviation of the predicted duration for this dataset?
# 
# - 1.24
# - 6.24
# - 12.28
# - 18.28

# In[12]:


# Find the standard deviation of the predicted duration
predicted_duration_std = y_pred.std()

print(
    f" The standard deviation of the predicted duration is {predicted_duration_std:.2f} minutes"
)

# The standard deviation of the predicted duration is 6.25 minutes
# So, answer is option B, 6.24 (closest to 6.25)


# ## Q2. Preparing the output 
# 
# Like in the course videos, we want to prepare the dataframe with the output.
# 
# First, let's create an artificial ride_id column:
# 
# `df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')`
# 
# Next, write the ride id and the predictions to a dataframe with results.
# 
# Save it as parquet:
# 
# ```python
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
# ```
# 
# What's the size of the output file?
# 
# - 36M
# - 46M
# - 56M
# - 66M
# 
# Note: Make sure you use the snippet above for saving the file. It should contain only these two columns. For this question, don't change the dtypes of the columns and use pyarrow, not fastparquet.

# In[13]:


# First, let's create an artificial ride_id column

year = 2023
month = 3

df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

# Next, write the ride id and the predictions to a dataframe with results.

results = pd.DataFrame({"ride_id": df["ride_id"], "predicted_duration": y_pred})

print(results.head())


# In[14]:


# Save it as parquet file

output_file = f"{year:04d}-{month:02d}-predictions.parquet"

results.to_parquet(output_file, engine="pyarrow", compression=None, index=False)

print(f"Predictions saved to {output_file}")


# In[15]:


# What is the size of the saved file (in MB)?

file_size = os.path.getsize(output_file) / (1024 * 1024)
print(f"Size of the saved file is {file_size:.2f} MB")

# Size of the saved file is 65.46 MB
# So, answer is option C, 66MB (closest to 65.46)


# ## Q3. Creating the scoring script
# 
# Now let's turn the notebook into a script.
# 
# Which command you need to execute for that?

# In[ ]:




