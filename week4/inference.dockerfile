# Base image
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Update pip 
RUN pip install --upgrade pip

# Install pipenv
RUN pip install pipenv

# Install pyarrow
# RUN pip install pyarrow

# Set the working directory
WORKDIR /app

# Copy the Pipfile and Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install the dependencies using pipenv
RUN pipenv install --system --deploy 

# Copy starter.py to the working directory
COPY ["week4/starter.py", "./"]

# Set the entrypoint (aka command to run when the container starts)
ENTRYPOINT ["python", "starter.py"]
