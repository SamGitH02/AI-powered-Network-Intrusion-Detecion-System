 # Use Python base image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy all files
# Copy the requirements file into the container
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy the application code into the container
COPY . .


# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]

