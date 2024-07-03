FROM python:slim

RUN apt-get update && apt-get -y install vim

ENV PIP_DISABLE_PIP_VERSION_CHECK=on

RUN pip install poetry==1.3.2

WORKDIR /app

COPY app app
COPY poetry.lock pyproject.toml analysis.py .
RUN poetry config virtualenvs.create false
RUN poetry lock --no-update
RUN poetry install --no-interaction

ENTRYPOINT /bin/bash
