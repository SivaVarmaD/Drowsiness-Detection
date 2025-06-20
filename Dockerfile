FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y cmake ffmpeg libsm6 libxext6

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
