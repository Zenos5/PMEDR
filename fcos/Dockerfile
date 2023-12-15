FROM pytorch/pytorch:latest

RUN apt update && apt install -y git nano vim

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

# CMD ./run_training.sh
