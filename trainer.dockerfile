# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_docker.txt requirements_docker.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements_docker.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

