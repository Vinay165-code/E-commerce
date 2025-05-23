# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "scalable_ecommerce_backend:app", "--host", "0.0.0.0", "--port", "8000"]
