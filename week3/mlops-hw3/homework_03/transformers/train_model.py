import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

@transformer 
def transform(df):
    # Assign categorical columns
    categorical_columns = ['PULocationID', 'DOLocationID']
    
    # Assign target column
    target = 'duration'
    
    # Convert categorical columns to category dtype
    for col in categorical_columns:
        df[col] = df[col].astype('category')
    
    # Separate features and target
    X = df[categorical_columns]
    y = df[target]
    
    # Create a DictVectorizer
    dv = DictVectorizer(sparse=True)
    
    # Transform categorical data to one-hot encoded features
    X_train = dv.fit_transform(X.to_dict(orient='records'))
    
    # Train a Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y)
    
    # Print the intercept of the model
    print("Intercept:", lr.intercept_)
    
    return dv, lr