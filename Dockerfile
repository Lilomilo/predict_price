from ubuntu:20.04 
MAINTAINER Lilomilo
RUN apt-get update -y
COPY . /opt/predict_price
WORKDIR /opt/predict_price
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 app_new.py
