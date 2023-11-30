FROM python:3.10

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /usr/src/app

# Creating Work Directory
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME
RUN mkdir static media

# Update pip
RUN pip install --upgrade pip

# Installing requirements.pip from project
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt

# Copy project to container
COPY ./src /usr/src/app
ADD . /usr/src/app/

