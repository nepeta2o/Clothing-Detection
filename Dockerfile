FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

RUN apt-get update && apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

ENV LANG=C.UTF-8
ENV FLASK_ENV=development
ENV FLASK_APP=app.py
EXPOSE 5002
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5002"]
