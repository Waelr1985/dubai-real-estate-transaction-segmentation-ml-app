# Use official Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies with heavy WSL timeout protection
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Set Streamlit healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the application
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
