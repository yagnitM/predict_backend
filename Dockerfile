# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by your project
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy your project files into the container
COPY . .

# Expose port 8000 for uvicorn
EXPOSE 8000

# Run the app with uvicorn when container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
