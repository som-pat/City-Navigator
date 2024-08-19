# Python base image 
FROM python:3.9-slim

#Set the working directory
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

CMD ["fastapi", "run", "app/main.py", "--port", "80"]


# docker build -t flask-app:v1.0 .
# docker run -d -p 80:80 flask-app:v1.0 
# docker container ls
# docker conatiner stop __