FROM python:slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY app app
COPY analysis.py KaggleV2-May-2016.csv ./

