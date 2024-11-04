FROM debian:stable-slim

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-dev build-essential python3-venv ffmpeg libopus-dev

RUN mkdir -p /code /file /logs /vect

ADD ./requirements.txt ./

RUN python3 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

ADD ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ADD . /code
WORKDIR /code

CMD ["bash"]
# END
